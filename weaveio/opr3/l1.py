import inspect
import sys

from weaveio.hierarchy import Optional, Multiple, Hierarchy
from weaveio.opr3.hierarchy import Spectrum, Single, FibreTarget, Stacked, \
    Stack, OB, ArmConfig, Superstack, OBSpec, Supertarget, WeaveTarget, RawSpectrum, _predicate, Spectrum1D, MeanFlux


class L1(Hierarchy):
    is_template = True


class L1Spectrum(Spectrum1D, L1):
    is_template = True
    children = [Optional('self', idname='adjunct')]
    products = ['flux', 'ivar', 'sensfunc']
    factors = Spectrum.factors + ['nspec', 'snr'] + MeanFlux.as_factors('g', 'r', 'i', 'gg', 'bp', 'rp')


class NoSS(Spectrum1D):
    plural_name = 'nosses'
    singular_name = 'noss'
    products = ['flux', 'ivar']
    parents = [L1Spectrum]
    children = [Optional('self', idname='adjunct')]
    identifier_builder = ['l1_spectrum']


class L1SingleSpectrum(L1Spectrum, Single):
    """
    A single spectrum row processed from a raw spectrum, belonging to one fibretarget and one run.
    """
    singular_name = 'l1single_spectrum'
    plural_name = 'l1single_spectra'
    parents = L1Spectrum.parents + [RawSpectrum, FibreTarget, ArmConfig]
    identifier_builder = ['raw_spectrum', 'fibre_target', 'arm_config']
    factors = L1Spectrum.factors + [
        'rms_arc1', 'rms_arc2', 'resol', 'helio_cor',
        'wave_cor1', 'wave_corrms1', 'wave_cor2', 'wave_corrms2',
        'skyline_off1', 'skyline_rms1', 'skyline_off2', 'skyline_rms2',
        'sky_shift', 'sky_scale']


class L1StackedSpectrum(L1Spectrum, Stacked):
    is_template = True
    singular_name = 'l1stack_spectrum'
    plural_name = 'l1stack_spectra'
    parents = [Multiple(L1SingleSpectrum, 2)]
    identifier_builder = ['l1single_spectra']


class L1StackSpectrum(L1StackedSpectrum, Stack):
    """
    A stacked spectrum row processed from > 1 single spectrum, belonging to one fibretarget but many runs within the same OB.
    """
    singular_name = 'l1stack_spectrum'
    plural_name = 'l1stack_spectra'
    parents = [Multiple(L1SingleSpectrum, 2, constrain=(OB, FibreTarget, ArmConfig))]


class L1SuperstackSpectrum(L1StackedSpectrum, Superstack):
    """
    A stacked spectrum row processed from > 1 single spectrum, belonging to one fibretarget but many runs within the same OBSpec.
    """
    singular_name = 'l1superstack_spectrum'
    plural_name = 'l1superstack_spectra'
    parents = [Multiple(L1SingleSpectrum, 2, constrain=(OBSpec, FibreTarget, ArmConfig))]


class L1SupertargetSpectrum(L1StackedSpectrum, Supertarget):
    """
    A stacked spectrum row processed from > 1 single spectrum, belonging to one weavetarget over many different OBSpecs.
    """
    singular_name = 'l1supertarget_spectrum'
    plural_name = 'l1supertarget_spectra'
    parents = [Multiple(L1SingleSpectrum, 2, constrain=(WeaveTarget, ArmConfig))]


hierarchies = [i[-1] for i in inspect.getmembers(sys.modules[__name__], _predicate)]