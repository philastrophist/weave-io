from pathlib import Path

import os

from weaveio.config_tables import progtemp_config
from weaveio.hierarchy import Hierarchy, Multiple, Indexed, One2One

HERE = Path(os.path.dirname(os.path.abspath(__file__)))


class Author(Hierarchy):
    is_template = True


class CASU(Author):
    """
    CASU is the pipeline that will produce the L1 and Raw data files and spectra.
    The version of CASU that produces these files may change without warning and files may be duplicated.
    This CASU object breaks that potential degeneracy.
    """
    idname = 'casuid'


class APS(Author):
    """
    APS is the pipeline that will produce the L2 data files and model spectra.
    The version of APS that produces these files may change without warning and files may be duplicated.
    This APS object breaks that potential degeneracy.
    """
    idname = 'apsvers'


class Simulator(Author):
    """
    Data which were simulated in an operation rehearsal will be be produced by a Simulator.
    Real data will not have a Simulator.
    """
    factors = ['simvdate', 'simver', 'simmode']
    identifier_builder = factors


class System(Author):
    idname = 'sysver'


class ArmConfig(Hierarchy):
    """
    An ArmConfig is the entire configuration of one arm of the WEAVE spectrograph.
    The ArmConfig is therefore a subset of the ProgTemp code and there are multiple ways to identify
    one:
    - resolution can be "high" or "low"
    - vph is the number of the grating [1=lowres, 2=highres, 3=highres(green)]
    - camera can be 'red' or 'blue'
    - colour can be 'red', 'blue', or 'green'
    """
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
    """
    Whilst ProgTemp deals with "how" a target is observed, OBSTEMP deals with "when" a target is observed,
    namely setting the observational constraints required to optimally extract scientific information from the observation.
    The ProgTemp is made up of maxseeing, mintrans (transparency), minelev (elevation), minmoon, maxsky and
    each combination is given a valid code, where each letter corresponds to one of those settings.
    """
    factors = ['maxseeing', 'mintrans', 'minelev', 'minmoon', 'maxsky', 'code']
    identifier_builder = factors[:-1]

    @classmethod
    def from_header(cls, header):
        names = [f.lower() for f in cls.factors[:-1]]
        obstemp_code = list(header['OBSTEMP'])
        return cls(**{n: v for v, n in zip(obstemp_code, names)}, code=header['OBSTEMP'])


class Survey(Hierarchy):
    """
    A Survey is one of the official Surveys of WEAVE. WL is one of them.
    """
    idname = 'name'


class WeaveTarget(Hierarchy):
    """
    A WeaveTarget is the disambiguated target to which all targets submitted by surveys belong.
    The "cname" of a weavetarget will be formed of the ra and dec of the submitted target.
    """
    idname = 'cname'


class Fibre(Hierarchy):
    """
    A WEAVE spectrograph fibre
    """
    idname = 'fibreid'


class SubProgramme(Hierarchy):
    """
    A submitted programme of observation which was written by multiple surveys.
    """
    parents = [Multiple(Survey)]
    factors = ['name']
    idname = 'progid'


class SurveyCatalogue(Hierarchy):
    """
    A catalogue which was submitted by a subprogramme.
    """
    parents = [SubProgramme]
    factors = ['name']
    idname = 'catid'


class SurveyTarget(Hierarchy):
    """
    A target which was submitted by a subprogramme contained within a catalogue. This is likely
    the target you want if you not linking observations between subprogrammes.
    """
    parents = [SurveyCatalogue, WeaveTarget]
    factors = ['targid', 'targname', 'targra', 'targdec', 'targepoch',
               'targpmra', 'targpmdec', 'targparal', 'mag_g', 'emag_g', 'mag_r', 'emag_r', 'mag_i', 'emag_i', 'mag_gg', 'emag_gg',
               'mag_bp', 'emag_bp', 'mag_rp', 'emag_rp']
    identifier_builder = ['weavetarget', 'surveycatalogue', 'targid', 'targra', 'targdec']


class InstrumentConfiguration(Hierarchy):
    """
    The WEAVE instrument can be configured into MOS/LIFU/mIFU modes and the spectral binning in pixels.
    InstrumentConfiguration holds the mode, binning, and is linked to an ArmConfig.
    """
    factors = ['mode', 'binning']
    parents = [Multiple(ArmConfig, 2, 2, idname='camera')]
    identifier_builder = ['armconfigs', 'mode', 'binning']


class ProgTemp(Hierarchy):
    """
    The ProgTemp code is an integral part of describing a WEAVE target.
    This parameter encodes the requested instrument configuration, OB length, exposure time,
    spectral binning, cloning requirements and probabilistic connection between these clones.
    The ProgTemp is therefore formed from instrumentconfiguration, length, and an exposure_code.
    """
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
    """
    When an xml observation specification is submitted to WEAVE, an OBSpec is created containing all
    the information about when and how to observe.
    When actually observing them, an "OB" is create with its own unique obid.
    """
    factors = ['obtitle']
    parents = [ObsTemp, ProgTemp, Multiple(SurveyCatalogue), Multiple(SubProgramme), Multiple(Survey)]
    idname = 'xml'  # this is CAT-NAME in the header not CATNAME, annoyingly no hyphens allowed


class FibreTarget(Hierarchy):
    """
    A fibretarget is the combination of fibre and surveytarget which is created after submission when
    the fibres are assigned.
    This object describes where the fibre is placed and what its status is.
    """
    factors = ['fibrera', 'fibredec', 'status', 'xposition', 'yposition',
               'orientat',  'retries', 'targx', 'targy', 'targuse', 'targprio']
    parents = [OBSpec, Fibre, SurveyTarget]
    identifier_builder = ['obspec', 'fibre', 'surveytarget', 'fibrera', 'fibredec', 'targuse']
    belongs_to = ['obspec', 'surveytarget']


class OB(Hierarchy):
    """
    An OB is an "observing block" which is essentially a realisation of an OBSpec.
    Many OBs can share the same xml OBSpec which describes how to do the observations.
    """
    idname = 'obid'  # This is globally unique by obid
    factors = ['obstartmjd']
    parents = [OBSpec]


class Exposure(Hierarchy):
    """
    An exposure is one observation of one set of targets for a given configuration.
    WEAVE is structured such that an exposure is actually two sets of data, one from each arm.
    These are called runs.
    """
    idname = 'expmjd'  # globally unique
    parents = [OB]


class Run(Hierarchy):
    """
    A run is one observation of a set of targets for a given configuration in a specific arm (red or blue).
    A run belongs to an exposure, which always consists of one or two runs (per arm).
    """
    idname = 'runid'
    parents = [ArmConfig, Exposure]


class Observation(Hierarchy):
    """
    A container for actual observing conditions around a run
    """
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
    """
    A 2D spectrum containing two counts arrays, this is not wavelength calibrated.
    """
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
    """
    A single spectrum row processed from a raw spectrum, belonging to one fibretarget and one run.
    """
    plural_name = 'l1singlespectra'
    parents = L1SpectrumRow.parents + [RawSpectrum, FibreTarget, CASU]
    version_on = ['rawspectrum', 'fibretarget']
    factors = L1SpectrumRow.factors + [
        'rms_arc1', 'rms_arc2', 'resol', 'helio_cor',
        'wave_cor1', 'wave_corrms1', 'wave_cor2', 'wave_corrms2',
        'skyline_off1', 'skyline_rms1', 'skyline_off2', 'skyline_rms2',
        'sky_shift', 'sky_scale']


class L1StackSpectrum(L1SpectrumRow):
    """
    A stacked spectrum row processed from > 1 single spectrum, belonging to one fibretarget but many runs within the same OB.
    """
    plural_name = 'l1stackspectra'
    parents = L1SpectrumRow.parents + [Multiple(L1SingleSpectrum, 2), OB, ArmConfig, FibreTarget, CASU]
    version_on = ['l1singlespectra', 'fibretarget']


class L1SuperStackSpectrum(L1SpectrumRow):
    """
    A stacked spectrum row processed from > 1 single spectrum, belonging to one fibretarget but many runs within the same OBSpec.
    """
    plural_name = 'l1superstackspectra'
    parents = L1SpectrumRow.parents + [Multiple(L1SingleSpectrum, 2), OBSpec, ArmConfig, FibreTarget, CASU]
    version_on = ['l1singlespectra', 'fibretarget']


class L1SuperTargetSpectrum(L1SpectrumRow):
    """
    A stacked spectrum row processed from > 1 single spectrum, belonging to one weavetarget over many different OBSpecs.
    """
    plural_name = 'l1supertargetspectra'
    parents = L1SpectrumRow.parents + [Multiple(L1SingleSpectrum, 2), WeaveTarget, CASU]
    version_on = ['l1singlespectra', 'weavetarget']


class L2(SourcedData):
    is_template = True


class L2Single(L2):
    """
    An L2 data product resulting from two or sometimes three single L1 spectra.
    The L2 data products contain information generated by APS namely redshifts, emission line properties and model spectra.
    """
    parents = [Multiple(L1SingleSpectrum, 2, 3), FibreTarget, APS, Exposure]


class L2Stack(L2):
    """
    An L2 data product resulting from two or sometimes three stacked/single L1 spectra.
    The L2 data products contain information generated by APS namely redshifts, emission line properties and model spectra.
    """
    parents = [Multiple(L1SingleSpectrum, 0, 3), Multiple(L1StackSpectrum, 0, 3), FibreTarget, APS, OB]


class L2SuperStack(L2):
    """
    An L2 data product resulting from two or sometimes three super-stacked/stacked/single L1 spectra.
    The L2 data products contain information generated by APS namely redshifts, emission line properties and model spectra.
    """
    parents = [Multiple(L1SingleSpectrum, 0, 3), Multiple(L1StackSpectrum, 0, 3), Multiple(L1SuperStackSpectrum, 0, 3), FibreTarget, APS, OBSpec]


class L2SuperTarget(L2):
    """
    An L2 data product resulting from two or sometimes three supertarget L1 spectra.
    The L2 data products contain information generated by APS namely redshifts, emission line properties and model spectra.
    """
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
    """
    The APS-generated table of propagated model fluxes, redshifts, and galaxy/star/qso classifications
    """
    factors = L2TableRow.factors + ['class', 'subclass', 'z', 'z_err', 'auto_class_alls',
                                    'auto_subclass_alls', 'z_alls', 'z_err_alls', 'rchi2diff',
                                    'rchi2_alls', 'rchi2diff_alls', 'zwarning', 'zwarning_alls',
                                    'sn_median_all', 'sn_medians', 'specflux_sloans',
                                    'specflux_sloan_ivars', 'specflux_johnsons',
                                    'specflux_johnson_ivars', 'specsynfluxes', 'specsynflux_ivars',
                                    'specskyflux']


class GalaxyTable(L2TableRow):
    """
    The APS-generated table of galaxy spectrum properties generated by GANDALF.
    """
    with open(HERE / 'galaxy_table_columns.txt', 'r') as _f:
        factors = L2TableRow.factors + [x.lower().strip() for x in _f.readlines() if len(x)]


class ClassificationSpectrum(L2Spectrum):
    """
    The joint-arm model spectrum used in the classification of this spectrum
    """
    plural_name = 'classification_spectra'
    products = {'flux': Indexed('class_spectra', 'flux'), 'ivar': Indexed('class_spectra', 'ivar'),
                'model': Indexed('class_spectra', 'model'), 'lambda': Indexed('class_spectra', 'lambda')}


class GalaxySpectrum(L2Spectrum):
    """
    The joint-arm model spectrum used in the GANDALF fitting.
    """
    plural_name = 'galaxy_spectra'
    products = {'flux': Indexed('galaxy_spectra', 'flux'), 'ivar': Indexed('galaxy_spectra', 'ivar'),
                'model_ab': Indexed('galaxy_spectra', 'model_ab'), 'model_em': Indexed('galaxy_spectra', 'model_em'),
                'lambda': Indexed('galaxy_spectra', 'lambda')}
