import inspect
import sys
import os
import pandas as pd
from pathlib import Path

from weaveio.hierarchy import Multiple, Hierarchy, OneOf, Optional
from weaveio.opr3.hierarchy import Spectrum, Author, APS, Measurement, \
    Single, FibreTarget, Exposure, Stack, OB, Superstack, \
    OBSpec, Supertarget, WeaveTarget, _predicate, MCMCMeasurement, Line, SpectralIndex, RedshiftMeasurement, Spectrum1D, ArrayHolder
from weaveio.opr3.l1 import L1Spectrum, L1SingleSpectrum, L1StackSpectrum, L1SupertargetSpectrum, L1SuperstackSpectrum, L1StackedSpectrum

HERE = Path(os.path.dirname(os.path.abspath(__file__)))
gandalf_lines = pd.read_csv(HERE / 'expected_lines.csv', sep=' ')
gandalf_indices = pd.read_csv(HERE / 'expected_line_indices.csv', sep=' ')
# gandalf_lines['name'] = gandalf_lines['name'].str.replace('[', '').str.replace(']', '').str.lower()
gandalf_line_names = (gandalf_lines['name'] + '_' + gandalf_lines['lambda'].apply(lambda x: f'{x:.2f}')).values.tolist()
gandalf_index_names = gandalf_indices['name'].values.tolist()


class L2(Hierarchy):
    is_template = True


class IngestedSpectrum(Spectrum1D):
    """
    An ingested spectrum is one which is a slightly modified version of an L1 spectrum
    """
    is_template = True
    children = []  # gets rid of inherited wvl array
    products = ['flux', 'error', 'wvl', 'ivar', 'logwvl', 'goodpix']


class UncombinedIngestedSpectrum(IngestedSpectrum):
    parents = [L1Spectrum]
    identifier_builder = ['l1_spectrum']


class CombinedIngestedSpectrum(IngestedSpectrum):
    parents = [Multiple(L1Spectrum, 1, 3)]
    identifier_builder = ['l1_spectra']


class ModelSpectrum(Spectrum1D):
    is_template = True
    children = []  # gets rid of inherited wvl array
    parents = [OneOf(IngestedSpectrum, one2one=True)]
    identifier_builder = ['ingested_spectrum']
    products = ['flux']


class UncombinedModelSpectrum(ModelSpectrum):
    parents = [OneOf(UncombinedIngestedSpectrum, one2one=True)]
    identifier_builder = ['uncombined_ingested_spectrum']


class BaseCombinedModelSpectrum(ModelSpectrum):
    is_template = True
    parents = [OneOf(CombinedIngestedSpectrum, one2one=True)]
    identifier_builder = ['combined_ingested_spectrum']


class CombinedModelSpectrum(BaseCombinedModelSpectrum):
    is_template = False


# This allows us to use 'clean' to talk about the clean model or clean spectrum
# IngestedSpectrum->ModelSpectrum->CleanModelSpectrum
# IngestedSpectrum->CleanIngestedSpectrum

class GandalfSpectrum(ModelSpectrum):
    """template object that only stores model flux and sources the wavelength array from another linked object"""
    is_template = True
    parents = []  # removes input spectrum dependency
    identifier_builder = None


class GandalfModelSpectrum(BaseCombinedModelSpectrum, GandalfSpectrum):
    """
    A Gandalf model spectrum is the full modelled SED of emission lines and continuum
    It also has:
        - `.clean_model`     : a clean model spectrum without emission line contribution
        - `.clean_observed`  : the observed spectrum with model emission lines subtracted
        - `.emission_line_model`  : the model emission line contribution only
    """

class GandalfDerivativeSpectrum(GandalfSpectrum):
    is_template = True
    parents = [OneOf(GandalfModelSpectrum, one2one=True)]
    identifier_builder = ['gandalf_model_spectrum']


class GandalfCleanIngestedSpectrum(GandalfDerivativeSpectrum):
    pass


class GandalfCleanModelSpectrum(GandalfDerivativeSpectrum):
    pass


class GandalfEmissionModelSpectrum(GandalfDerivativeSpectrum):
    pass


class Fit(Hierarchy):
    """
    A fit is the result of applying fitting_software to an ingested spectrum
    In the case of combined spectra being available, there is only one ingested spectrum input
    otherwise, there are more.
    """
    is_template = True
    parents = [Multiple(L1Spectrum, 2, 3, one2one=True),
               Multiple(UncombinedModelSpectrum, 0, 3, one2one=True),
               Optional(BaseCombinedModelSpectrum, one2one=True),
               Multiple(ModelSpectrum, 1, 3, one2one=True, notreal=True)]


class RedshiftArray(ArrayHolder):
    factors = ['value', 'start', 'end', 'step']
    identifier_builder = ['start', 'end', 'step']


class Template(Fit):
    parents = [Multiple(L1Spectrum, 2, 3),
               Multiple(UncombinedModelSpectrum, 0, 3, one2one=True), Optional(CombinedModelSpectrum, one2one=True),
               Multiple(ModelSpectrum, 1, 3, one2one=True, notreal=True)]
    factors = ['chi2_array', 'name']
    children = [OneOf(RedshiftArray, one2one=True)]
    identifier_builder = ['uncombined_model_spectra', 'combined_model_spectrum', 'name']


class Redrock(Fit):
    factors = ['class', 'subclass', 'snr', 'chi2', 'deltachi2', 'ncoeff', 'coeff',
               'npixels', 'srvy_class', 'z', 'zerr', 'zwarn']
    parents = [Multiple(L1Spectrum, 2, 3),
               Multiple(UncombinedModelSpectrum, 0, 3, one2one=True), Optional(CombinedModelSpectrum, one2one=True),
               Multiple(ModelSpectrum, 1, 3, one2one=True, notreal=True)]
    # todo: really need to have a check in the merge that checks that combinedmodelspec doesnt exist if not given
    template_names = ['galaxy', 'qso', 'star_a', 'star_b', 'star_cv', 'star_f', 'star_g', 'star_k', 'star_m', 'star_wd']
    parents += [OneOf(Template, idname=x, one2one=True) for x in template_names]
    identifier_builder = ['uncombined_model_spectra', 'combined_model_spectrum']


class RVSpecfit(Fit):
    singular_name = 'rvspecfit'
    parents = [Multiple(L1Spectrum, 2, 3),
               Multiple(UncombinedModelSpectrum, 0, 3, one2one=True), Optional(CombinedModelSpectrum, one2one=True),
               Multiple(ModelSpectrum, 1, 3, one2one=True, notreal=True)]
    factors = Fit.factors + ['skewness', 'kurtosis', 'vsini', 'snr', 'chi2_tot']
    factors += Measurement.as_factors('vrad', 'logg', 'teff', 'feh', 'alpha')
    identifier_builder = ['uncombined_model_spectra', 'combined_model_spectrum']


class Ferre(Fit):
    parents = [Multiple(L1Spectrum, 2, 3),
               Multiple(UncombinedModelSpectrum, 0, 3, one2one=True), Optional(CombinedModelSpectrum, one2one=True),
               Multiple(ModelSpectrum, 1, 3, one2one=True, notreal=True)]
    factors = Fit.factors + ['snr', 'chi2_tot', 'flag']
    factors += Measurement.as_factors('micro', 'logg', 'teff', 'feh', 'alpha', 'elem')
    identifier_builder = ['uncombined_model_spectra', 'combined_model_spectrum']


class Gandalf(Fit):
    plural_name = 'gandalfs'
    parents = [Multiple(L1Spectrum, 2, 3),
               OneOf(GandalfModelSpectrum, one2one=True, idname='model'),
               OneOf(GandalfCleanIngestedSpectrum, one2one=True, idname='clean_observed'),
               OneOf(GandalfCleanModelSpectrum, one2one=True, idname='clean_model'),
               OneOf(GandalfEmissionModelSpectrum, one2one=True, idname='emission_line_model'),
               ]
    factors = Fit.factors + ['fwhm_flag']
    factors += Line.as_factors(*gandalf_line_names) + SpectralIndex.as_factors(*gandalf_index_names)
    identifier_builder = ['model']


class PPXF(Fit):
    parents = [Multiple(L1Spectrum, 2, 3), OneOf(CombinedModelSpectrum, one2one=True)]
    factors = Fit.factors + MCMCMeasurement.as_factors('v', 'sigma', 'h3', 'h4', 'h5', 'h6') + \
              Measurement.as_factors('zcorr')
    identifier_builder = ['combined_model_spectrum']


class L2Product(L2):
    is_template = True
    parents = [Multiple(L1Spectrum, 2, 3),
               Optional(Redrock, one2one=True), Optional(RVSpecfit, one2one=True),
               Optional(Ferre, one2one=True), Optional(PPXF, one2one=True), Optional(Gandalf, one2one=True)]


# L2 data products are formed from 2 or more L1 data products from different arms (red, blue, or green)
# L2 singles can only be formed from 2 single L1 data products
# Since an OB has a fixed instrument configuration, L2 stacks can only be formed from 2 L1 stacks
# However, APS tries to create the widest and deepest data possible, so L2 superstacks are not limit in their L1 spectra provenance

class L2Single(L2Product, Single):
    """
    An L2 data product resulting from two or sometimes three single L1 spectra.
    The L2 data products contain information generated by APS namely redshifts, emission line properties and model spectra.

    """
    singular_name = 'l2single'
    parents = L2Product.parents[1:] + [Multiple(L1SingleSpectrum, 2, 2, constrain=(FibreTarget, Exposure))]
    identifier_builder = ['l1single_spectra', 'fibre_target', 'exposure']


class L2Stack(L2Product, Stack):
    """
    An L2 data product resulting from two or sometimes three stacked/single L1 spectra.
    The L2 data products contain information generated by APS namely redshifts, emission line properties and model spectra.
    """
    singular_name = 'l2stack'
    parents = L2Product.parents[1:] + [Multiple(L1StackSpectrum, 2, 2, constrain=(FibreTarget, OB))]
    identifier_builder = ['l1stack_spectra', 'fibre_target', 'ob']


class L2Superstack(L2Product, Superstack):
    """
    An L2 data product resulting from two or sometimes three super-stacked/stacked/single L1 spectra.
    The L2 data products contain information generated by APS namely redshifts, emission line properties and model spectra.
    """
    singular_name = 'l2superstack'
    parents = L2Product.parents[1:] + [Multiple(L1StackedSpectrum, 2, 3, constrain=(FibreTarget, OBSpec))]
    identifier_builder = ['l1stacked_spectra', 'fibre_target', 'obspec']


class L2Supertarget(L2Product, Supertarget):
    """
    An L2 data product resulting from two or sometimes three supertarget L1 spectra.
    The L2 data products contain information generated by APS namely redshifts, emission line properties and model spectra.
    """
    singular_name = 'l2supertarget'
    parents = L2Product.parents[1:] + [Multiple(L1SupertargetSpectrum, 2, 3, constrain=(WeaveTarget,))]
    identifier_builder = ['l1supertarget_spectra', 'weave_target']


hierarchies = [i[-1] for i in inspect.getmembers(sys.modules[__name__], _predicate)]
