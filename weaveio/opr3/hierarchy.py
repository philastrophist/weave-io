from pathlib import Path

import os

from weaveio.config_tables import progtemp_config
from weaveio.file import File
from weaveio.hierarchy import Hierarchy, Multiple, Indexed


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
    idname = 'surveyname'


class WeaveTarget(Hierarchy):
    idname = 'cname'


class Fibre(Hierarchy):
    idname = 'fibreid'


class SubProgramme(Hierarchy):
    parents = [Multiple(Survey)]
    idname = 'targprog'


class SurveyCatalogue(Hierarchy):
    parents = [SubProgramme]
    idname = 'targcat'


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
    parents = [ObsTemp, ProgTemp]
    idname = 'xml'  # this is CAT-NAME in the header not CATNAME, annoyingly no hyphens allowed


class FibreTarget(Hierarchy):
    factors = ['fibrera', 'fibredec', 'status', 'xposition', 'yposition',
               'orientat',  'retries', 'targx', 'targy', 'targuse', 'targprio']
    parents = [OBSpec, Fibre, SurveyTarget]
    identifier_builder = ['obspec', 'fibre', 'surveytarget', 'fibrera', 'fibredec', 'targuse']
    # belongs_to = [OBSpec]


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
    parents = [Run, CASU, Simulator, System]
    factors = ['mjdobs', 'seeing', 'windspb', 'windspe', 'humidb', 'humide', 'winddir', 'airpres', 'tempb', 'tempe', 'skybrght', 'observer']
    products = ['primary', 'guidinfo', 'metinfo']
    identifier_builder = ['run', 'mjdobs']
    version_on = ['run']

    @classmethod
    def from_header(cls, run, header):
        factors = {f: header.get(f) for f in cls.factors}
        factors['mjdobs'] = header['MJD-OBS']
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
    parents = [Observation, CASU]
    factors = ['sourcefile']
    identifier_builder = ['sourcefile']
    products = ['counts1', 'counts2']
    version_on = ['observation']
    # any duplicates under a run will be versioned based on their appearance in the database
    # only one raw per run essentially


class L1SpectrumRow(Spectrum):
    plural_name = 'l1spectrumrows'
    is_template = True
    products = ['primary', Indexed('flux'), Indexed('ivar'), Indexed('flux_noss'), Indexed('ivar_noss'), 'sensfunc']


class L1SingleSpectrum(L1SpectrumRow):
    plural_name = 'l1singlespectra'
    parents = L1SpectrumRow.parents + [RawSpectrum, FibreTarget, CASU]
    version_on = ['rawspectrum', 'fibretarget']
    factors = L1SpectrumRow.factors + [
        'nspec', 'rms_arc1', 'rms_arc2', 'resol', 'helio_cor',
        'wave_cor1', 'wave_corrms1', 'wave_cor2', 'wave_corrms2',
        'skyline_off1', 'skyline_rms1', 'skyline_off2', 'skyline_rms2',
        'sky_shift', 'sky_scale', 'exptime', 'snr',
        'meanflux_g', 'meanflux_r', 'meanflux_i',
        'meanflux_gg', 'meanflux_bp', 'meanflux_rp'
               ]


class L1StackSpectrum(L1SpectrumRow):
    plural_name = 'l1stackspectra'
    parents = L1SpectrumRow.parents + [Multiple(L1SingleSpectrum, 2), OB, ArmConfig, FibreTarget, CASU]
    version_on = ['l1singlespectra', 'fibretarget']
    factors = L1SpectrumRow.factors + ['nspec', 'exptime', 'snr', 'meanflux_g', 'meanflux_r', 'meanflux_i',
               'meanflux_gg', 'meanflux_bp', 'meanflux_rp']


class L1SuperStackSpectrum(L1SpectrumRow):
    plural_name = 'l1superstackspectra'
    parents = L1SpectrumRow.parents + [Multiple(L1SingleSpectrum, 2), OBSpec, ArmConfig, FibreTarget, CASU]
    factors = L1SpectrumRow.factors + ['nspec', 'exptime', 'snr', 'meanflux_g', 'meanflux_r', 'meanflux_i',
               'meanflux_gg', 'meanflux_bp', 'meanflux_rp']
    version_on = ['l1singlespectra', 'fibretarget']


class L1SuperTargetSpectrum(L1SpectrumRow):
    plural_name = 'l1supertargetspectra'
    parents = L1SpectrumRow.parents + [Multiple(L1SingleSpectrum, 2), WeaveTarget, CASU]
    factors = L1SpectrumRow.factors + ['nspec', 'exptime', 'snr', 'meanflux_g', 'meanflux_r', 'meanflux_i',
               'meanflux_gg', 'meanflux_bp', 'meanflux_rp']
    version_on = ['l1singlespectra', 'weavetarget']


class L2(SourcedData):
    is_template = True
    parents = [Multiple(L1SpectrumRow, 2, 3), FibreTarget, APS]
    version_on = ['l1spectrumrows']


class StackL2(L2):
    is_template = True
    parents = L2.parents + [OB]


class SuperStackL2(L2):
    is_template = True
    parents = L2.parents + [OBSpec]


L2_FTYPES = [StackL2, SuperStackL2]


class L2TableRow(L2):
    is_template = True


class L2Spectrum(Spectrum, L2):
    is_template = True
    plural_name = 'l2spectra'


class ClassificationTable(L2TableRow):
    is_template = True
    factors = L2TableRow.factors + ['class', 'subclass', 'z', 'z_err', 'auto_class_alls',
                                    'auto_subclass_alls', 'z_alls', 'z_err_alls', 'rchi2diff',
                                    'rchi2_alls', 'rchi2diff_alls', 'zwarning', 'zwarning_alls',
                                    'sn_median_all', 'sn_medians', 'specflux_sloans',
                                    'specflux_sloan_ivars', 'specflux_johnsons',
                                    'specflux_johnson_ivars', 'specsynfluxes', 'specsynflux_ivars',
                                    'specskyflux']


class GalaxyTable(L2TableRow):
    is_template = True
    with open(HERE / 'galaxy_table_columns', 'r') as _f:
        factors = L2TableRow.factors + [x.lower().strip() for x in _f.readlines() if len(x)]


class ClassificationSpectrum(L2Spectrum):
    plural_name = 'classification_spectra'
    is_template = True
    products = [Indexed('class_spectra', 'flux'), Indexed('class_spectra', 'ivar'), Indexed('class_spectra', 'model'),
                Indexed('class_spectra', 'lambda')]


class GalaxySpectrum(L2Spectrum):
    plural_name = 'galaxy_spectra'
    is_template = True
    products = [Indexed('galaxy_spectra', 'flux'), Indexed('galaxy_spectra', 'ivar'),
                Indexed('galaxy_spectra', 'model_ab'), Indexed('galaxy_spectra', 'model_em'),
                Indexed('galaxy_spectra', 'lambda')]


L2_DTYPES = [ClassificationTable, GalaxyTable, ClassificationSpectrum, GalaxySpectrum]


