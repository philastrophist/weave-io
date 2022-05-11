import inspect
import sys
import os
import pandas as pd
from pathlib import Path

from weaveio.hierarchy import Multiple, Hierarchy, OneOf
from weaveio.opr3.hierarchy import SourcedData, Spectrum, Author, APS, Measurement, \
    Single, FibreTarget, Exposure, OBStack, OB, Superstack, \
    OBSpec, Supertarget, WeaveTarget, _predicate, MCMCMeasurement, Line, SpectralIndex, RedshiftMeasurement, Spectrum1D, ArrayHolder
from weaveio.opr3.l1 import L1Spectrum, L1SingleSpectrum, L1OBStackSpectrum, L1SupertargetSpectrum, L1SuperstackSpectrum, L1StackSpectrum

HERE = Path(os.path.dirname(os.path.abspath(__file__)))
gandalf_lines = pd.read_csv(HERE / 'expected_lines.csv', sep=' ')
gandalf_indices = pd.read_csv(HERE / 'expected_line_indices.csv', sep=' ')
gandalf_lines['name'] = gandalf_lines['name'].str.replace('[', '').str.replace(']', '').str.lower()
gandalf_line_names = (gandalf_lines['name'] + '_' + gandalf_lines['lambda'].apply(lambda x: f'{x:.0f}')).values.tolist()
gandalf_index_names = gandalf_indices['name'].values.tolist()


class L2(SourcedData):
    is_template = True


class IngestedSpectrum(Spectrum1D):
    """
    An ingested spectrum is one which is a slightly modified version of an L1 spectrum
    """
    is_template = True
    factors = ['sourcefile', 'nrow']
    parents = [Multiple(L1Spectrum, 1, 3), APS]
    identifier_builder = ['sourcefile', 'nrow', 'l1_spectra']

class RestFrameIngestedSpectrum(IngestedSpectrum):
    is_template = True

class RedrockIngestedSpectrum(IngestedSpectrum):
    products = [ 'flux', 'ivar', 'wvl']

class RVSpecFitIngestedSpectrum(IngestedSpectrum):
    singular_name = 'rvspecfit_ingested_spectrum'
    products = [ 'flux', 'error', 'wvl']

class FerreIngestedSpectrum(IngestedSpectrum):
    products = [ 'flux', 'error', 'wvl']

class PPXFIngestedSpectrum(RestFrameIngestedSpectrum):
    products = [ 'flux', 'error', 'logwvl', 'goodpix']

class GandalfIngestedSpectrum(RestFrameIngestedSpectrum):
    products = [ 'flux', 'error', 'logwvl', 'goodpix']



class ModelSpectrum(Spectrum):
    is_template = True
    factors = ['sourcefile', 'nrow']
    identifier_builder = ['sourcefile', 'nrow']
    parents = [Multiple(IngestedSpectrum, 1, 3, one2one=True), Multiple(L1Spectrum, 1, 3, one2one=True)]
    products = ['flux', 'wvl']


class RedrockModelSpectrum(ModelSpectrum):
    parents = [Multiple(RedrockIngestedSpectrum, 1, 4), Multiple(L1Spectrum, 1, 3, one2one=True)]


class RVSpecfitModelSpectrum(ModelSpectrum):
    singular_name = 'rvspecfit_model_spectrum'
    parents = [Multiple(RVSpecFitIngestedSpectrum, 1, 4), Multiple(L1Spectrum, 1, 3, one2one=True)]


class FerreModelSpectrum(ModelSpectrum):
    parents = [Multiple(FerreIngestedSpectrum, 1, 4), Multiple(L1Spectrum, 1, 3, one2one=True)]


class PPXFModelSpectrum(ModelSpectrum):
    parents = [PPXFIngestedSpectrum, Multiple(L1Spectrum, 1, 3, one2one=True)]


class LogarithmicModelSpectrum(ModelSpectrum):
    products = ['logwvl', 'flux']


class GandalfModelSpectrum(LogarithmicModelSpectrum):
    parents = [GandalfIngestedSpectrum, Multiple(L1Spectrum, 1, 3, one2one=True)]
    children = [OneOf(LogarithmicModelSpectrum, idname='clean', one2one=True),
                OneOf(LogarithmicModelSpectrum, idname='emission', one2one=True)]


class FittingSoftware(Author):
    idname = 'version'


class Fit(Hierarchy):
    """
    A fit is the result of applying fitting_software to an ingested spectrum
    In the case of combined spectra being available, there is only one ingested spectrum input
    otherwise, there are more.
    """
    is_template = True
    parents = [FittingSoftware, Multiple(IngestedSpectrum, 1, 3, one2one=True), Multiple(L1Spectrum, 1, 3, one2one=True)]
    children = [ModelSpectrum]


class RedrockVersion(FittingSoftware):
    pass

class RVSpecfitVersion(FittingSoftware):
    singular_name = 'rvspecfit_version'

class FerreVersion(FittingSoftware):
    pass

class PPXFVersion(FittingSoftware):
    pass

class GandalfVersion(FittingSoftware):
    pass


class RedshiftArray(ArrayHolder):
    idname = 'hash'
    factors = ['value']


class RedrockTemplate(Hierarchy):
    parents = [OneOf(RedshiftArray, one2one=True)]
    factors = ['chi2_array', 'name']
    indexes = ['name']


class Redrock(Fit):
    factors = Fit.factors + ['flag', 'class', 'subclass', 'snr', 'best_chi2', 'deltachi2', 'ncoeff', 'coeff',
                             'npixels', 'srvy_class'] + RedshiftMeasurement.as_factors('best_redshift')
    parents = [RedrockVersion, Multiple(RedrockIngestedSpectrum, 1, 4, one2one=True),
               Multiple(L1Spectrum, 1, 3, one2one=True)]
    children = [OneOf(RedrockTemplate, idname=x, one2one=True) for x in ['galaxy', 'qso', 'star_a', 'star_b', 'star_cv', 'star_f', 'star_g', 'star_k', 'star_m', 'star_wd']]
    children += [RedrockModelSpectrum]
    identifier_builder = ['redrock_version', 'redrock_ingested_spectra', 'snr']


class RVSpecfit(Fit):
    singular_name = 'rvspecfit'
    parents = [RVSpecfitVersion, Multiple(RVSpecFitIngestedSpectrum, 1, 4, one2one=True),
               Multiple(L1Spectrum, 1, 3, one2one=True)]
    factors = Fit.factors + ['skewness', 'kurtosis', 'vsini', 'snr', 'chi2_tot']
    factors += Measurement.as_factors('vrad', 'logg', 'teff', 'feh', 'alpha')
    children = [RVSpecfitModelSpectrum]
    identifier_builder = ['rvspecfit_version', 'rvspecfit_ingested_spectra', 'snr']


class Ferre(Fit):
    parents = [FerreVersion, Multiple(FerreIngestedSpectrum, 1, 4, one2one=True),
               Multiple(L1Spectrum, 1, 3, one2one=True)]
    factors = Fit.factors + ['snr', 'chi2_tot', 'flag']
    factors += Measurement.as_factors('micro', 'logg', 'teff', 'feh', 'alpha', 'elem')
    children = [FerreModelSpectrum]
    identifier_builder = ['ferre_version', 'ferre_ingested_spectra', 'snr']


class Gandalf(Fit):
    parents = [GandalfVersion, OneOf(GandalfIngestedSpectrum, one2one=True),
               Multiple(L1Spectrum, 1, 3, one2one=True)]
    factors = Fit.factors + ['fwhm_flag'] + Measurement.as_factors('zcorr')
    factors += Line.as_factors(gandalf_line_names) + SpectralIndex.as_factors(gandalf_index_names)
    children = [GandalfModelSpectrum]
    identifier_builder = ['gandalf_version', 'gandalf_ingested_spectrum', 'zcorr']


class PPXF(Fit):
    parents = [PPXFVersion, OneOf(PPXFIngestedSpectrum, one2one=True),
               Multiple(L1Spectrum, 1, 3, one2one=True)]
    factors = Fit.factors + MCMCMeasurement.as_factors('v', 'sigma', 'h3', 'h4', 'h5', 'h6')
    children = [PPXFModelSpectrum]
    identifier_builder = ['ppxf_version', 'ppxf_ingested_spectrum', 'v']


class L2Product(L2):
    is_template = True
    parents = [Multiple(L1Spectrum, 2, 3), APS]
    children = [Multiple(RedrockIngestedSpectrum, 1, 4), Multiple(RVSpecFitIngestedSpectrum, 1, 4),
                Multiple(FerreIngestedSpectrum, 1, 4),
                PPXFIngestedSpectrum, GandalfIngestedSpectrum,
                Redrock, RVSpecfit, Ferre, PPXF, Gandalf]


# L2 data products are formed from 2 or more L1 data products from different arms (red, blue, or green)
# L2 singles can only be formed from 2 single L1 data products
# Since an OB has a fixed instrument configuration, L2 obstacks can only be formed from 2 L1 obstacks
# However, APS tries to create the widest and deepest data possible, so L2 superstacks are not limit in their L1 spectra provenance

class L2Single(L2Product, Single):
    """
    An L2 data product resulting from two or sometimes three single L1 spectra.
    The L2 data products contain information generated by APS namely redshifts, emission line properties and model spectra.

    """
    singular_name = 'l2single'
    parents = [Multiple(L1SingleSpectrum, 2, 2, constrain=(FibreTarget, Exposure), one2one=True), APS]


class L2OBStack(L2Product, OBStack):
    """
    An L2 data product resulting from two or sometimes three stacked/single L1 spectra.
    The L2 data products contain information generated by APS namely redshifts, emission line properties and model spectra.
    """
    singular_name = 'l2obstack'
    parents = [Multiple(L1OBStackSpectrum, 2, 2, constrain=(FibreTarget, OB), one2one=True), APS]


class L2Superstack(L2Product, Superstack):
    """
    An L2 data product resulting from two or sometimes three super-stacked/stacked/single L1 spectra.
    The L2 data products contain information generated by APS namely redshifts, emission line properties and model spectra.
    """
    singular_name = 'l2superstack'
    parents = [Multiple(L1StackSpectrum, 2, 3, constrain=(FibreTarget, OBSpec)), APS]


class L2Supertarget(L2Product, Supertarget):
    """
    An L2 data product resulting from two or sometimes three supertarget L1 spectra.
    The L2 data products contain information generated by APS namely redshifts, emission line properties and model spectra.
    """
    singular_name = 'l2supertarget'
    parents = [Multiple(L1SupertargetSpectrum, 2, 3, constrain=(WeaveTarget,), one2one=True), APS]


hierarchies = [i[-1] for i in inspect.getmembers(sys.modules[__name__], _predicate)]
