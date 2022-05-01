import inspect
import sys

from weaveio.hierarchy import Optional, Indexed, Multiple, Hierarchy
from weaveio.opr3.hierarchy import Spectrum, Single, FibreTarget, Stack, \
    OBStack, OB, ArmConfig, Superstack, OBSpec, Supertarget, WeaveTarget, RawSpectrum, _predicate


class L1(Hierarchy):
    is_template = True


class NoSS(Spectrum, L1):
    plural_name = 'nosses'
    singular_name = 'noss'
    products = {'flux': Indexed('flux'), 'ivar': Indexed('ivar')}
    children = [Optional('self', idname='adjunct')]


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


class WavelengthHolder(Hierarchy):
    factors = ['wvl', 'cd1_1', 'crval1', 'naxis1']
    identifier_builder = ['cd1_1', 'crval1', 'naxis1']


class MeanFlux(Hierarchy):
    singular_name = 'mean_flux'
    plural_name = 'mean_fluxes'
    idname = 'band'
    factors = ['value']


hierarchies = [i[-1] for i in inspect.getmembers(sys.modules[__name__], _predicate)]