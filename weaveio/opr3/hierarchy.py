from pathlib import Path

import os

from weaveio.config_tables import progtemp_config
from weaveio.hierarchy import Hierarchy, Multiple, Indexed, One2One

HERE = Path(os.path.dirname(os.path.abspath(__file__)))


class Author(Hierarchy):
    is_template = True


class CASU(Author):
    idname = 'casuid'


class APS(Author):
    idname = 'apsvers'


class Simulator(Author):
    factors = ['simvdate', 'simver', 'simmode']
    identifier_builder = factors


class System(Author):
    idname = 'sysver'


class ArmConfig(Hierarchy):
    factors = ['resolution', 'vph', 'camera', 'colour']
    identifier_builder = ['resolution', 'vph', 'camera']

    def __init__(self, tables=None, **kwargs):
        if kwargs['vph'] == 3 and kwargs['camera'] == 'blue':
            kwargs['colour'] = 'green'
        else:
            kwargs['colour'] = kwargs['camera']
        super().__init__(tables, **kwargs)

    @classmethod
    def from_progtemp_code(cls, progtemp_code):
        config = progtemp_config.loc[progtemp_code[0]]
        red = cls(resolution=str(config.resolution), vph=int(config.red_vph), camera='red')
        blue = cls(resolution=str(config.resolution), vph=int(config.blue_vph), camera='blue')
        return red, blue


class ObsTemp(Hierarchy):
    factors = ['maxseeing', 'mintrans', 'minelev', 'minmoon', 'maxsky', 'code']
    identifier_builder = factors[:-1]

    @classmethod
    def from_header(cls, header):
        names = [f.lower() for f in cls.factors[:-1]]
        obstemp_code = list(header['OBSTEMP'])
        return cls(**{n: v for v, n in zip(obstemp_code, names)}, code=header['OBSTEMP'])


class Survey(Hierarchy):
    idname = 'name'


class WeaveTarget(Hierarchy):
    idname = 'cname'


class Fibre(Hierarchy):
    idname = 'fibreid'


class SubProgramme(Hierarchy):
    parents = [Multiple(Survey)]
    factors = ['name']
    idname = 'progid'


class SurveyCatalogue(Hierarchy):
    parents = [SubProgramme]
    factors = ['name']
    idname = 'catid'


class SurveyTarget(Hierarchy):
    parents = [SurveyCatalogue, WeaveTarget]
    factors = ['targid', 'targname', 'targra', 'targdec', 'targepoch',
               'targpmra', 'targpmdec', 'targparal', 'mag_g', 'emag_g', 'mag_r', 'emag_r', 'mag_i', 'emag_i', 'mag_gg', 'emag_gg',
               'mag_bp', 'emag_bp', 'mag_rp', 'emag_rp']
    identifier_builder = ['weavetarget', 'surveycatalogue', 'targid', 'targra', 'targdec']


class InstrumentConfiguration(Hierarchy):
    factors = ['mode', 'binning']
    parents = [Multiple(ArmConfig, 2, 2, idname='camera')]
    identifier_builder = ['armconfigs', 'mode', 'binning']


class ProgTemp(Hierarchy):
    parents = [InstrumentConfiguration]
    factors = ['length', 'exposure_code', 'code']
    identifier_builder = ['instrumentconfiguration'] + factors

    @classmethod
    def from_progtemp_code(cls, progtemp_code):
        progtemp_code = progtemp_code.split('.')[0]
        progtemp_code_list = list(map(int, progtemp_code))
        configs = ArmConfig.from_progtemp_code(progtemp_code_list)
        mode = progtemp_config.loc[progtemp_code_list[0]]['mode']
        binning = progtemp_code_list[3]
        config = InstrumentConfiguration(armconfigs=configs, mode=mode, binning=binning)
        exposure_code = progtemp_code[2:4]
        length = progtemp_code_list[1]
        return cls(code=progtemp_code, length=length, exposure_code=exposure_code,
                   instrumentconfiguration=config)


class OBSpec(Hierarchy):
    factors = ['obtitle']
    parents = [ObsTemp, ProgTemp, Multiple(SurveyCatalogue), Multiple(SubProgramme), Multiple(Survey)]
    idname = 'xml'  # this is CAT-NAME in the header not CATNAME, annoyingly no hyphens allowed


class FibreTarget(Hierarchy):
    factors = ['fibrera', 'fibredec', 'status', 'xposition', 'yposition',
               'orientat',  'retries', 'targx', 'targy', 'targuse', 'targprio']
    parents = [OBSpec, Fibre, SurveyTarget]
    identifier_builder = ['obspec', 'fibre', 'surveytarget', 'fibrera', 'fibredec', 'targuse']
    belongs_to = ['obspec', 'surveytarget']


class OB(Hierarchy):
    idname = 'obid'  # This is globally unique by obid
    factors = ['obstartmjd']
    parents = [OBSpec]


class Exposure(Hierarchy):
    idname = 'expmjd'  # globally unique
    parents = [OB]


class Run(Hierarchy):
    idname = 'runid'
    parents = [ArmConfig, Exposure]


class Observation(Hierarchy):
    parents = [One2One(Run), CASU, Simulator, System]
    factors = ['mjdobs', 'seeing', 'windspb', 'windspe', 'humidb', 'humide', 'winddir', 'airpres', 'tempb', 'tempe', 'skybrght', 'observer']
    products = {'primary': 'primary', 'guidinfo': 'guidinfo', 'metinfo': 'metinfo'}
    identifier_builder = ['run', 'mjdobs']
    version_on = ['run']

    @classmethod
    def from_header(cls, run, header):
        factors = {f: header.get(f) for f in cls.factors}
        factors['mjdobs'] = float(header['MJD-OBS'])
        casu = CASU(casuid=header.get('casuvers', header.get('casuid')))
        sim = Simulator(simver=header['simver'], simmode=header['simmode'], simvdate=header['simvdate'])
        sys = System(sysver=header['sysver'])
        return cls(run=run, casu=casu, simulator=sim, system=sys, **factors)


class SourcedData(Hierarchy):
    is_template = True
    factors = ['sourcefile', 'nrow']
    identifier_builder = ['sourcefile', 'nrow']


class Spectrum(SourcedData):
    is_template = True
    plural_name = 'spectra'


class RawSpectrum(Spectrum):
    plural_name = 'rawspectra'
    parents = [One2One(Observation), CASU]
    products = {'counts1': 'counts1', 'counts2': 'counts2'}
    version_on = ['observation']
    # any duplicates under a run will be versioned based on their appearance in the database
    # only one raw per run essentially


class WavelengthHolder(Hierarchy):
    factors = ['wvls', 'cd1_1', 'crval1', 'naxis1']
    identifier_builder = ['cd1_1', 'crval1', 'naxis1']


class L1SpectrumRow(Spectrum):
    plural_name = 'l1spectrumrows'
    is_template = True
    products = {'primary': 'primary', 'flux': Indexed('flux'), 'ivar': Indexed('ivar'),
                'flux_noss': Indexed('flux_noss'), 'ivar_noss': Indexed('ivar_noss'), 'sensfunc': 'sensfunc'}
    factors = Spectrum.factors + ['nspec', 'exptime', 'snr', 'meanflux_g', 'meanflux_r', 'meanflux_i', 'meanflux_gg', 'meanflux_bp', 'meanflux_rp']


class L1SingleSpectrum(L1SpectrumRow):
    plural_name = 'l1singlespectra'
    parents = L1SpectrumRow.parents + [RawSpectrum, FibreTarget, CASU]
    version_on = ['rawspectrum', 'fibretarget']
    factors = L1SpectrumRow.factors + [
        'rms_arc1', 'rms_arc2', 'resol', 'helio_cor',
        'wave_cor1', 'wave_corrms1', 'wave_cor2', 'wave_corrms2',
        'skyline_off1', 'skyline_rms1', 'skyline_off2', 'skyline_rms2',
        'sky_shift', 'sky_scale']


class L1StackSpectrum(L1SpectrumRow):
    plural_name = 'l1stackspectra'
    parents = L1SpectrumRow.parents + [Multiple(L1SingleSpectrum, 2), OB, ArmConfig, FibreTarget, CASU]
    version_on = ['l1singlespectra', 'fibretarget']


class L1SuperStackSpectrum(L1SpectrumRow):
    plural_name = 'l1superstackspectra'
    parents = L1SpectrumRow.parents + [Multiple(L1SingleSpectrum, 2), OBSpec, ArmConfig, FibreTarget, CASU]
    version_on = ['l1singlespectra', 'fibretarget']


class L1SuperTargetSpectrum(L1SpectrumRow):
    plural_name = 'l1supertargetspectra'
    parents = L1SpectrumRow.parents + [Multiple(L1SingleSpectrum, 2), WeaveTarget, CASU]
    version_on = ['l1singlespectra', 'weavetarget']


class L2(SourcedData):
    is_template = True


class L2Single(L2):
    parents = [Multiple(L1SingleSpectrum, 2, 3), FibreTarget, APS, Exposure]


class L2Stack(L2):
    parents = [Multiple(L1SingleSpectrum, 0, 3), Multiple(L1StackSpectrum, 0, 3), FibreTarget, APS, OB]


class L2SuperStack(L2):
    parents = [Multiple(L1SingleSpectrum, 0, 3), Multiple(L1StackSpectrum, 0, 3), Multiple(L1SuperStackSpectrum, 0, 3), FibreTarget, APS, OBSpec]


class L2SuperTarget(L2):
    parents = [Multiple(L1SuperTargetSpectrum, 2, 3), APS, WeaveTarget]


class L2SourcedData(Hierarchy):
    is_template = True
    factors = ['sourcefile', 'hduname', 'nrow']
    identifier_builder = ['sourcefile', 'hduname', 'nrow']
    parents = [One2One(L2)]
    belongs_to = ['l2']


class L2TableRow(L2SourcedData):
    is_template = True


class L2Spectrum(L2SourcedData):
    is_template = True
    plural_name = 'l2spectra'
    products = {'flux': Indexed('*_spectra', 'flux'), 'ivar': Indexed('*_spectra', 'ivar'),
                'model_ab': Indexed('*_spectra', 'model_ab'), 'model_em': Indexed('*_spectra', 'model_em'),
                'lambda': Indexed('*_spectra', 'lambda')}


class ClassificationTable(L2TableRow):
    factors = L2TableRow.factors + ['class', 'subclass', 'z', 'z_err', 'auto_class_alls',
                                    'auto_subclass_alls', 'z_alls', 'z_err_alls', 'rchi2diff',
                                    'rchi2_alls', 'rchi2diff_alls', 'zwarning', 'zwarning_alls',
                                    'sn_median_all', 'sn_medians', 'specflux_sloans',
                                    'specflux_sloan_ivars', 'specflux_johnsons',
                                    'specflux_johnson_ivars', 'specsynfluxes', 'specsynflux_ivars',
                                    'specskyflux']


class GalaxyTable(L2TableRow):
    with open(HERE / 'galaxy_table_columns.txt', 'r') as _f:
        factors = L2TableRow.factors + [x.lower().strip() for x in _f.readlines() if len(x)]


class ClassificationSpectrum(L2Spectrum):
    plural_name = 'classification_spectra'
    products = {'flux': Indexed('class_spectra', 'flux'), 'ivar': Indexed('class_spectra', 'ivar'),
                'model': Indexed('class_spectra', 'model'), 'lambda': Indexed('class_spectra', 'lambda')}


class GalaxySpectrum(L2Spectrum):
    plural_name = 'galaxy_spectra'
    products = {'flux': Indexed('galaxy_spectra', 'flux'), 'ivar': Indexed('galaxy_spectra', 'ivar'),
                'model_ab': Indexed('galaxy_spectra', 'model_ab'), 'model_em': Indexed('galaxy_spectra', 'model_em'),
                'lambda': Indexed('galaxy_spectra', 'lambda')}
