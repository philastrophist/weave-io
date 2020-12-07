from weaveio.config_tables import progtemp_config
from weaveio.hierarchy import Hierarchy, Multiple


class Author(Hierarchy):
    is_template = True


class CASU(Author):
    idname = 'version'


class APS(Author):
    idname = 'version'


class Simulator(Author):
    idname = 'version'
    factors = ['simvdate', 'simver', 'simmode']


class System(Author):
    idname = 'version'


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
    factors = ['maxseeing', 'mintrans', 'minelev', 'minmoon', 'maxsky']
    identifier_builder = factors

    @classmethod
    def from_header(cls, header):
        names = [f.lower() for f in cls.factors]
        obstemp_code = list(header['OBSTEMP'])
        return cls(**{n: v for v, n in zip(obstemp_code, names)})


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
    factors = ['length', 'exposure_code']
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
        return cls(progtemp_code=progtemp_code, length=length, exposure_code=exposure_code,
                   instrumentconfiguration=config)


class FibreTarget(Hierarchy):
    factors = ['fibrera', 'fibredec', 'status', 'xposition', 'yposition',
               'orientat',  'retries', 'targx', 'targy', 'targuse', 'targprio']
    parents = [Fibre, SurveyTarget]
    identifier_builder = ['fibre', 'surveytarget', 'fibrera', 'fibredec', 'targuse']


class OBSpec(Hierarchy):
    factors = ['obtitle']
    parents = [ObsTemp, ProgTemp, Multiple(FibreTarget, 0)]
    idname = 'xml'  # this is CAT-NAME in the header not CATNAME, annoyingly no hyphens allowed


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


class Spectrum(Hierarchy):
    plural_name = 'spectra'
    is_template = True
    idname = 'hashid'
    products = ['flux', 'ivar', 'noss_flux', 'noss_ivar']


class RawSpectrum(Spectrum):
    plural_name = 'rawspectra'
    parents = [Run, CASU, Simulator, System]
    products = Spectrum.products + ['guideinfo', 'metinfo']
    version_on = ['run']
    # any duplicates under a run will be versioned based on their appearance in the database
    # only one raw per run essentially


class L1SpectrumRow(Spectrum):
    plural_name = 'l1spectrumrows'
    is_template = True


class L1SingleSpectrum(L1SpectrumRow):
    plural_name = 'l1singlespectra'
    parents = [RawSpectrum, FibreTarget, CASU]
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
    parents = [Multiple(L1SingleSpectrum, 2), OB, ArmConfig, FibreTarget, CASU]
    version_on = ['l1singlespectra', 'fibretarget']
    factors = L1SpectrumRow.factors + ['exptime', 'snr', 'meanflux_g', 'meanflux_r', 'meanflux_i',
               'meanflux_gg', 'meanflux_bp', 'meanflux_rp']


class L1SuperStackSpectrum(L1SpectrumRow):
    plural_name = 'l1superstackspectra'
    parents = [Multiple(L1SingleSpectrum, 2), OBSpec, ArmConfig, FibreTarget, CASU]
    factors = ['exptime', 'snr', 'meanflux_g', 'meanflux_r', 'meanflux_i',
               'meanflux_gg', 'meanflux_bp', 'meanflux_rp']
    version_on = ['l1singlespectra']


class L1SuperTargetSpectrum(L1SpectrumRow):
    plural_name = 'l1supertargetspectra'
    parents = [Multiple(L1SingleSpectrum, 2), WeaveTarget, CASU]
    factors = ['exptime', 'snr', 'meanflux_g', 'meanflux_r', 'meanflux_i',
               'meanflux_gg', 'meanflux_bp', 'meanflux_rp']
    version_on = ['l1singlespectra']


class L2(Hierarchy):
    is_template = True


class L2RowHDU(L2):
    is_template = True
    parents = [Multiple(L1SpectrumRow, 2, 3), APS]
    products = []
    version_on = ['l1spectrumrows']


# class Classifications(L2RowHDU):
#     pass
#
#
# class Star(L2RowHDU):
#     pass
#
#
# class Galaxy(L2RowHDU):
#     pass
#
#
# class L2Spectrum(L2RowHDU):
#     is_template = True
#
#
# class ClassificationModelSpectrum(L2Spectrum):
#     pass
#
#
# class StellarModelSpectrum(L2Spectrum):
#     pass
