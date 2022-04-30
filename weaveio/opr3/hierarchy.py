"""
Hierarchies are defined like so:

class Name(Type):
    singular_name = 'custom_singular_name'
    plural_name = 'custom_plural_name'
    idname = 'the_name_of_the_unique_attribute_of_this_Object'
    identifier_builder = [list_of_attributes_or_objects_that_define_uniqueness]
    factors = ['attribute1', 'attribute2']
    products = {'name_of_product': product_object}
    parents = [required_object_that_comes_before]
    children = [required_object_that_comes_after]

if an object of type O requires n parents of type P then this is equivalent to defining that instances of those behave as:
    parent-(n)->object (1 object has n parents of type P)
    it implicitly follows that:
        object--(m)-parent (each of object's parents of type P can be used by an unknown number `m` of objects of type O = many to one)
if an object of type O requires n children of type C then this is equivalent to defining that instances of those behave as:
    object-(n)->child (1 object has n children of type C)
    it implicitly follows that:
        child-[has 1]->Object (each child has maybe 1 parent of type O)

A child does not need to require a parent at instantiation

You can modify the first relation in each of the above by using:
    Multiple(parent/child, minnumber, maxnumber)
    One2One(parent/child) - makes a reciprocal one to one relationship
    Optional(parent/child) == Multiple(parent/Child, 0, 1)

"""
from pathlib import Path

import os

import pandas as pd

from weaveio.config_tables import progtemp_config
from weaveio.hierarchy import Hierarchy, Multiple, Indexed, Optional, OneOf

HERE = Path(os.path.dirname(os.path.abspath(__file__)))
gandalf_line_data = pd.read_csv(HERE / 'expected_lines.csv', sep=' ')
gandalf_index_data = pd.read_csv(HERE / 'expected_line_indices.csv', sep=' ')


class Measurement(Hierarchy):
    is_template = True
    factors = ['value', 'error']


class Author(Hierarchy):
    is_template = True


class CASU(Author):
    """
    CASU is the pipeline that will produce the L1 and Raw data files and spectra.
    The version of CASU that produces these files may change without warning and files may be duplicated.
    This CASU object breaks that potential degeneracy.
    """
    idname = 'id'


class APS(Author):
    """
    APS is the pipeline that will produce the L2 data files and model spectra.
    The version of APS that produces these files may change without warning and files may be duplicated.
    This APS object breaks that potential degeneracy.
    """
    idname = 'version'


class Simulator(Author):
    """
    Data which were simulated in an operation rehearsal will be be produced by a Simulator.
    Real data will not have a Simulator.
    """
    factors = ['date', 'version', 'mode']
    identifier_builder = factors


class System(Author):
    idname = 'version'


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
    The ProgTemp is made up of maxseeing, mintrans (transparency), minelev (elevation), minmoon, minsky and
    each combination is given a valid code, where each letter corresponds to one of those settings.
    """
    factors = [
        'maxseeing', 'mintrans', 'minelev', 'minmoon', 'minsky', 'maxairmass',
        'maxseeing_grade', 'mintrans_grade', 'minelev_grade', 'minmoon_grade', 'minsky_grade',
    ]
    idname = 'code'
    seeings = {'A': 0.7, 'B': 0.8, 'C': 0.9, 'D': 1.0, 'E': 1.1, 'F': 1.2, 'G': 1.3, 'H': 1.4, 'I': 1.5,
               'J': 1.6, 'K': 1.7, 'L': 1.8, 'M': 1.9, 'N': 2.0, 'O': 2.1, 'P': 2.2, 'Q': 2.3, 'R': 2.4,
               'S': 2.5, 'T': 2.6, 'U': 2.7, 'V': 2.8, 'W': 2.9, 'X': 3.0}
    transparencies = {'A': 0.8, 'B': 0.7, 'C': 0.6, 'D': 0.5, 'E': 0.4}
    elevs = {'A': 50.28, 'B': 45.58, 'C': 41.81, 'D': 35.68, 'E': 33.75, 'F': 25.00}
    airmasses = {'A': 1.3, 'B': 1.4, 'C': 1.5, 'D': 1.6, 'E': 1.8, 'F': 2.4}
    moons = {'A': 90, 'B': 70, 'C': 50, 'D': 30, 'E': 0}
    brightnesses = {'A': 21.7, 'B': 21.5, 'C': 21.0, 'D': 20.5, 'E': 19.6, 'F': 18.5, 'G': 17.7}

    @classmethod
    def from_header(cls, header):
        names = [f.lower() for f in cls.factors[:-1]]
        obstemp_code = list(header['OBSTEMP'])
        data = {n+'_grade': v for v, n in zip(obstemp_code, names)}
        data['maxseeing'] = cls.seeings[data['maxseeing_grade']]
        data['mintrans'] = cls.transparencies[data['mintrans_grade']]
        data['minelev'] = cls.elevs[data['minelev_grade']]
        data['maxairmass'] = cls.airmasses[data['minelev_grade']]
        data['minmoon'] = cls.moons[data['minmoon_grade']]
        data['minsky'] = cls.brightnesses[data['minsky_grade']]
        return cls(code=header['OBSTEMP'], **data)


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
    idname = 'id'


class SubProgramme(Hierarchy):
    """
    A submitted programme of observation which was written by multiple surveys.
    """
    parents = [Multiple(Survey)]
    factors = ['name']
    idname = 'id'


class SurveyCatalogue(Hierarchy):
    """
    A catalogue which was submitted by a subprogramme.
    """
    parents = [SubProgramme]
    factors = ['name']
    idname = 'id'


class Magnitude(Measurement):
    idname = 'band'


class SurveyTarget(Hierarchy):
    """
    A target which was submitted by a subprogramme contained within a catalogue. This is likely
    the target you want if you not linking observations between subprogrammes.
    """
    parents = [SurveyCatalogue, WeaveTarget]
    factors = ['id', 'name', 'ra', 'dec', 'epoch', 'pmra', 'pmdec', 'paral']
    children = Multiple.from_names(Magnitude, 'g', 'r', 'i', 'gg', 'bp', 'rp')
    identifier_builder = ['weave_target', 'survey_catalogue', 'id', 'ra', 'dec']


class InstrumentConfiguration(Hierarchy):
    """
    The WEAVE instrument can be configured into MOS/LIFU/mIFU modes and the spectral binning in pixels.
    InstrumentConfiguration holds the mode, binning, and is linked to an ArmConfig.
    """
    factors = ['mode', 'binning']
    parents = [Multiple(ArmConfig, 2, 2)]
    identifier_builder = ['arm_configs', 'mode', 'binning']


class ProgTemp(Hierarchy):
    """
    The ProgTemp code is an integral part of describing a WEAVE target.
    This parameter encodes the requested instrument configuration, OB length, exposure time,
    spectral binning, cloning requirements and probabilistic connection between these clones.
    The ProgTemp is therefore formed from instrumentconfiguration, length, and an exposure_code.
    """
    parents = [InstrumentConfiguration]
    factors = ['length', 'exposure_code', 'code']
    identifier_builder = ['instrument_configuration'] + factors

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


class FibreTarget(Hierarchy):
    """
    A fibretarget is the combination of fibre and surveytarget which is created after submission when
    the fibres are assigned.
    This object describes where the fibre is placed and what its status is.
    """
    factors = ['ra', 'dec', 'status', 'orientation', 'nretries', 'x', 'y', 'use', 'priority']
    parents = [Fibre, SurveyTarget]
    identifier_builder = ['fibre', 'survey_target', 'ra', 'dec', 'use']


class OBSpec(Hierarchy):
    """
    When an xml observation specification is submitted to WEAVE, an OBSpec is created containing all
    the information about when and how to observe.
    When actually observing them, an "OB" is create with its own unique obid.
    """
    singular_name = 'obspec'
    factors = ['title']
    parents = [ObsTemp, ProgTemp, Multiple(FibreTarget), Multiple(SurveyCatalogue), Multiple(SubProgramme), Multiple(Survey)]
    idname = 'xml'  # this is CAT-NAME in the header not CATNAME, annoyingly no hyphens allowed


class OB(Hierarchy):
    """
    An OB is an "observing block" which is essentially a realisation of an OBSpec.
    Many OBs can share the same xml OBSpec which describes how to do the observations.
    """
    idname = 'id'  # This is globally unique by obid
    factors = ['mjd']
    parents = [OBSpec]


class Exposure(Hierarchy):
    """
    An exposure is one observation of one set of targets for a given configuration.
    WEAVE is structured such that an exposure is actually two sets of data, one from each arm.
    These are called runs.
    """
    idname = 'mjd'  # globally unique
    factors = ['exptime', 'seeing', 'windspb', 'windspe', 'humidb', 'humide', 'winddir', 'airpres',
               'tempb', 'tempe', 'skybrght', 'observer', 'obstype']
    parents = [OB, CASU, Simulator, System]


    @classmethod
    def from_header(cls, ob, header):
        factors = {f: header.get(f) for f in cls.factors}
        factors['mjd'] = float(header['MJD-OBS'])
        factors['obstype'] = header['obstype'].lower()
        casu = CASU(id=header.get('casuvers', header.get('casuid')))
        try:
            sim = Simulator(simver=header['simver'], simmode=header['simmode'], simvdate=header['simvdate'])
        except KeyError:
            sim = None
        sys = System(sysver=header['sysver'])
        return cls(ob=ob, casu=casu, simulator=sim, system=sys, **factors)


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
    products = {'counts1': 'counts1', 'counts2': 'counts2'}
    # only one raw per run essentially


class Run(Hierarchy):
    """
    A run is one observation of a set of targets for a given configuration in a specific arm (red or blue).
    A run belongs to an exposure, which always consists of one or two runs (per arm).
    """
    idname = 'id'
    parents = [ArmConfig, Exposure]
    children = [RawSpectrum]


class WavelengthHolder(Hierarchy):
    factors = ['wvl', 'cd1_1', 'crval1', 'naxis1']
    identifier_builder = ['cd1_1', 'crval1', 'naxis1']


class MeanFlux(Hierarchy):
    singular_name = 'mean_flux'
    plural_name = 'mean_fluxes'
    idname = 'band'
    factors = ['value']


class L1(Hierarchy):
    is_template = True


class NoSS(Spectrum, L1):
    plural_name = 'nosses'
    singular_name = 'noss'
    products = {'flux': Indexed('flux'), 'ivar': Indexed('ivar')}
    children = [Optional('self', idname='adjunct')]


class Single(Hierarchy):
    is_template = True


class Stack(Hierarchy):
    is_template = True


class OBStack(Stack):
    is_template = True


class Superstack(Stack):
    is_template = True


class Supertarget(Stack):
    is_template = True


class L1SpectrumRow(Spectrum, L1):
    singular_name = 'l1_spectrum_row'
    plural_name = 'l1_spectrum_rows'
    is_template = True
    children = [Optional('self', idname='adjunct'), NoSS] # + Multiple.from_names(MeanFlux, 'g', 'r', 'i', 'gg', 'bp', 'rp')

    products = {'primary': 'primary', 'flux': Indexed('flux'), 'ivar': Indexed('ivar'),
                'sensfunc': Indexed('sensfunc'), 'wvl': 'wvl'}
    factors = Spectrum.factors + ['nspec', 'snr']


class L1SingleSpectrum(L1SpectrumRow, Single):
    """
    A single spectrum row processed from a raw spectrum, belonging to one fibretarget and one run.
    """
    singular_name = 'l1single_spectrum'
    plural_name = 'l1single_spectra'
    parents = L1SpectrumRow.parents + [RawSpectrum, FibreTarget]
    version_on = ['raw_spectrum', 'fibre_target']
    factors = L1SpectrumRow.factors + [
        'rms_arc1', 'rms_arc2', 'resol', 'helio_cor',
        'wave_cor1', 'wave_corrms1', 'wave_cor2', 'wave_corrms2',
        'skyline_off1', 'skyline_rms1', 'skyline_off2', 'skyline_rms2',
        'sky_shift', 'sky_scale']


class L1StackSpectrum(L1SpectrumRow, Stack):
    is_template = True
    singular_name = 'l1stack_spectrum'
    plural_name = 'l1stack_spectra'


class L1OBStackSpectrum(L1StackSpectrum, OBStack):
    """
    A stacked spectrum row processed from > 1 single spectrum, belonging to one fibretarget but many runs within the same OB.
    """
    singular_name = 'l1obstack_spectrum'
    plural_name = 'l1obstack_spectra'
    parents = [Multiple(L1SingleSpectrum, 2, constrain=(OB, FibreTarget, ArmConfig))]
    version_on = ['l1single_spectra', 'fibre_target']


class L1SuperstackSpectrum(L1StackSpectrum, Superstack):
    """
    A stacked spectrum row processed from > 1 single spectrum, belonging to one fibretarget but many runs within the same OBSpec.
    """
    singular_name = 'l1superstack_spectrum'
    plural_name = 'l1superstack_spectra'
    parents =[Multiple(L1SingleSpectrum, 2, constrain=(OBSpec, FibreTarget, ArmConfig))]
    version_on = ['l1single_spectra', 'fibre_target']


class L1SupertargetSpectrum(L1SpectrumRow, Supertarget):
    """
    A stacked spectrum row processed from > 1 single spectrum, belonging to one weavetarget over many different OBSpecs.
    """
    singular_name = 'l1supertarget_spectrum'
    plural_name = 'l1supertarget_spectra'
    parents = [Multiple(L1SingleSpectrum, 2, constrain=(WeaveTarget, ArmConfig))]
    version_on = ['l1single_spectra', 'weave_target']


class L2(SourcedData):
    is_template = True


class L2Single(L2, Single):
    """
    An L2 data product resulting from two or sometimes three single L1 spectra.
    The L2 data products contain information generated by APS namely redshifts, emission line properties and model spectra.
    """
    singular_name = 'l2single'
    parents = [Multiple(L1SingleSpectrum, 2, 3, constrain=(FibreTarget, Exposure)), APS]


class L2OBStack(L2, OBStack):
    """
    An L2 data product resulting from two or sometimes three stacked/single L1 spectra.
    The L2 data products contain information generated by APS namely redshifts, emission line properties and model spectra.
    """
    singular_name = 'l2stack'
    parents = [Multiple(L1OBStackSpectrum, 2, 3, constrain=(FibreTarget, OB)), APS]


class L2SuperStack(L2, Superstack):
    """
    An L2 data product resulting from two or sometimes three super-stacked/stacked/single L1 spectra.
    The L2 data products contain information generated by APS namely redshifts, emission line properties and model spectra.
    """
    singular_name = 'l2superstack'
    parents = [Multiple(L1SpectrumRow, 2, 3, constrain=(FibreTarget, OBSpec)), APS]


class L2SuperTarget(L2, Supertarget):
    """
    An L2 data product resulting from two or sometimes three supertarget L1 spectra.
    The L2 data products contain information generated by APS namely redshifts, emission line properties and model spectra.
    """
    singular_name = 'l2supertarget'
    parents = [Multiple(L1SupertargetSpectrum, 2, 3, constrain=(WeaveTarget,)), APS]


class L2Spectrum(Spectrum, L2):
    is_template = True
    factors = ['sourcefile', 'hduname', 'nrow']
    identifier_builder = ['sourcefile', 'hduname', 'nrow']
    parents = [L2]
    singular_name = 'l2spectrum'
    plural_name = 'l2spectra'
    products = {'flux': Indexed('*_spectra', 'flux'), 'ivar': Indexed('*_spectra', 'ivar'), 'lambda': Indexed('*_spectra', 'lambda')}


class IngestedL2Spectrum(L2Spectrum):
    singular_name = 'ingested_l2spectrum'
    plural_name = 'ingested_l2spectra'
    products = {'flux': Indexed('*_spectra', 'flux'), 'ivar': Indexed('*_spectra', 'ivar'), 'lambda': Indexed('*_spectra', 'lambda')}


class L2ModelSpectrum(Spectrum, L2):
    is_template = True


class L2SpectrumLogLam(L2Spectrum):
    products = {'flux': Indexed('*_spectra', 'flux'), 'err': Indexed('*_spectra', 'err'),
                'model': Indexed('*_spectra', 'model'), 'goodpix': Indexed('*_spectra', 'goodpix'),
                'loglambda': Indexed('*_spectra', 'loglam')}


class GandalfL2Spectrum(L2SpectrumLogLam):
    products = {
        'flux': Indexed('*_spectra', 'flux'), 'err': Indexed('*_spectra', 'err'),
        'flux_clean': Indexed('*_spectra', 'flux_clean'), 'model_clean': Indexed('*_spectra', 'model_clean'),
        'emission': Indexed('*spectra', 'emission'),
        'model': Indexed('*_spectra', 'model'), 'goodpix': Indexed('*_spectra', 'goodpix'),
        'loglambda': Indexed('*_spectra', 'loglam')
    }


class Fitter(Author):
    factors = ['version', 'name']
    identifier_builder = ['version', 'name']


class Fit(Hierarchy):
    is_template = True
    parents = [IngestedL2Spectrum, APS, Fitter]
    identifier_builder = ['aps', 'fitter', 'ingested_l2spectrum']


class MCMCMeasurement(Measurement):
    is_template = True
    factors = ['value', 'mcmc_error', 'formal_error']


class Line(Measurement):
    idname = 'name'
    factors = ['wvl', 'aon', 'vaccum']
    children = Multiple.from_names(Measurement, 'flux', 'redshift', 'sigma', 'ebmv', 'amp')


class SpectralIndex(Measurement):
    idname = 'name'


class RedshiftMeasurement(Measurement):
    indexes = ['value']
    factors = Measurement.factors + ['warn']


class RedshiftGrid(Hierarchy):
    # we make this a different object because it avoids replicating the same grid every time
    idname = 'hash'
    factors = ['value']


class RedrockTemplate(Fit):
    children = [OneOf(RedshiftGrid, idname='redshifts')]
    factors = ['name', 'chi2s']
    identifier_builder = ['aps', 'fitter', 'ingested_l2spectrum', 'name']


class RedrockFit(Fit):
    factors = Fit.factors + ['flag', 'class', 'subclass', 'snr', 'best_chi2', 'deltachi2', 'ncoeff', 'coeff',
                             'npixels', 'srvy_class']
    children = [OneOf.from_names(RedrockTemplate, 'galaxy', 'qso', 'star_a', 'star_b', 'star_cv',
                                 'star_f', 'star_g', 'star_k', 'star_m', 'star_wd')]
    children += [L2ModelSpectrum, OneOf(RedshiftMeasurement, idname='best_redshift')]


class RVSpecFit(Fit):
    factors = Fit.factors + ['skewness', 'kurtosis', 'vsini', 'snr', 'chi2_tot']
    children = Multiple.from_names(Measurement, 'vrad', 'logg', 'teff', 'feh', 'alpha')


class FerreFit(Fit):
    factors = Fit.factors + ['snr', 'chi2_tot', 'flag']
    children = Multiple.from_names(Measurement, 'micro', 'logg', 'teff', 'feh', 'alpha', 'elem')


class GandalfFit(Fit):
    children = [Multiple(Line), Multiple(SpectralIndex)] + Multiple.from_names(Measurement, 'zcorr')
    factors = Fit.factors + ['fwhm_flag']


class PPXFFit(Fit):
    children = Multiple.from_names(MCMCMeasurement, 'v', 'sigma', 'h3', 'h4', 'h5', 'h6')


import sys, inspect
hierarchies = list(filter(lambda x: issubclass(x, Hierarchy), [obj for name, obj in inspect.getmembers(sys.modules[__name__]) if inspect.isclass(obj)]))
