import inspect
import sys
import os
import pandas as pd
from pathlib import Path

from weaveio.hierarchy import Multiple, Hierarchy, OneOf
from weaveio.opr3.hierarchy import SourcedData, Spectrum, Author, APS, Measurement, \
    Single, FibreTarget, Exposure, OBStack, OB, Superstack, \
    OBSpec, Supertarget, WeaveTarget, _predicate, MCMCMeasurement, Line, SpectralIndex, RedshiftMeasurement, Spectrum1D
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


class RedrockVersion(FittingSoftware):
    pass

class RVSpecFitVersion(FittingSoftware):
    singular_name = 'rvspecfit_version'

class FerreVersion(FittingSoftware):
    pass

class PPXFVersion(FittingSoftware):
    pass

class GandalfVersion(FittingSoftware):
    pass


class RedrockTemplate(Hierarchy):
    is_template = True
    factors = ['redshifts', 'chi2s']


class RedrockFit(Fit):
    factors = Fit.factors + ['flag', 'class', 'subclass', 'snr', 'best_chi2', 'deltachi2', 'ncoeff', 'coeff',
                             'npixels', 'srvy_class'] + RedshiftMeasurement.as_factors('best_redshift')
    factors += RedrockTemplate.as_factors('galaxy', 'qso', 'star_a', 'star_b', 'star_cv',
                                 'star_f', 'star_g', 'star_k', 'star_m', 'star_wd')
    parents = [RedrockVersion, Multiple(RedrockIngestedSpectrum, 1, 4, one2one=True),
               Multiple(L1Spectrum, 1, 3, one2one=True)]
    identifier_builder = ['redrock_version', 'redrock_ingested_spectra', 'snr']


class RVSpecFit(Fit):
    singular_name = 'rvspecfit'
    parents = [RVSpecFitVersion, Multiple(RVSpecFitIngestedSpectrum, 1, 4, one2one=True),
               Multiple(L1Spectrum, 1, 3, one2one=True)]
    factors = Fit.factors + ['skewness', 'kurtosis', 'vsini', 'snr', 'chi2_tot']
    factors += Measurement.as_factors('vrad', 'logg', 'teff', 'feh', 'alpha')
    identifier_builder = ['rvspecfit_version', 'rvspecfit_ingested_spectra', 'snr']


class FerreFit(Fit):
    parents = [FerreVersion, Multiple(FerreIngestedSpectrum, 1, 4, one2one=True),
               Multiple(L1Spectrum, 1, 3, one2one=True)]
    factors = Fit.factors + ['snr', 'chi2_tot', 'flag']
    factors += Measurement.as_factors('micro', 'logg', 'teff', 'feh', 'alpha', 'elem')
    identifier_builder = ['ferre_version', 'ferre_ingested_spectra', 'snr']


class GandalfFit(Fit):
    parents = [GandalfVersion, OneOf(GandalfIngestedSpectrum, one2one=True),
               Multiple(L1Spectrum, 1, 3, one2one=True)]
    factors = Fit.factors + ['fwhm_flag'] + Measurement.as_factors('zcorr')
    factors += Line.as_factors(gandalf_line_names) + SpectralIndex.as_factors(gandalf_index_names)
    identifier_builder = ['gandalf_version', 'gandalf_ingested_spectrum', 'zcorr']


class PPXFFit(Fit):
    parents = [PPXFVersion, OneOf(PPXFIngestedSpectrum, one2one=True),
               Multiple(L1Spectrum, 1, 3, one2one=True)]
    factors = Fit.factors + MCMCMeasurement.as_factors('v', 'sigma', 'h3', 'h4', 'h5', 'h6')
    identifier_builder = ['ppxf_version', 'ppxf_ingested_spectrum', 'v']


class L2ModelSpectrum(Spectrum):
    is_template = True
    factors = ['sourcefile', 'nrow']
    identifier_builder = ['sourcefile', 'nrow']
    parents = [Fit, Multiple(IngestedSpectrum, 1, 3, one2one=True), Multiple(L1Spectrum, 1, 3, one2one=True)]
    products = ['model', 'wvl']


class RedrockModelSpectrum(L2ModelSpectrum):
    parents = [RedrockFit, Multiple(RedrockIngestedSpectrum, 1, 4), Multiple(L1Spectrum, 1, 3, one2one=True)]


class RVSpecFitModelSpectrum(L2ModelSpectrum):
    parents = [RVSpecFit, Multiple(RVSpecFitIngestedSpectrum, 1, 4), Multiple(L1Spectrum, 1, 3, one2one=True)]


class FerreModelSpectrum(L2ModelSpectrum):
    parents = [FerreFit, Multiple(FerreIngestedSpectrum, 1, 4), Multiple(L1Spectrum, 1, 3, one2one=True)]


class PPXFModelSpectrum(L2ModelSpectrum):
    parents = [PPXFFit, PPXFIngestedSpectrum, Multiple(L1Spectrum, 1, 3, one2one=True)]


class CompositeModelSpectrum(L2ModelSpectrum):
    is_template = True


class GandalfModelSpectrum(CompositeModelSpectrum):
    parents = [GandalfFit, GandalfIngestedSpectrum, Multiple(L1Spectrum, 1, 3, one2one=True)]
    products = ['model', 'logwvl', 'clean', 'emission', 'flux_clean']


class L2Product(L2):
    is_template = True
    parents = [Multiple(L1Spectrum, 2, 3), APS]
    children = [Multiple(RedrockIngestedSpectrum, 1, 4), Multiple(RVSpecFitIngestedSpectrum, 1, 4),
                Multiple(FerreIngestedSpectrum, 1, 4),
                PPXFIngestedSpectrum, GandalfIngestedSpectrum,
                RedrockFit, RVSpecFit, FerreFit, PPXFFit, GandalfFit]


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
    parents = [Multiple(L1StackSpectrum, 2, 3, constrain=(FibreTarget, )), APS]


class L2Supertarget(L2Product, Supertarget):
    """
    An L2 data product resulting from two or sometimes three supertarget L1 spectra.
    The L2 data products contain information generated by APS namely redshifts, emission line properties and model spectra.
    """
    singular_name = 'l2supertarget'
    parents = [Multiple(L1SupertargetSpectrum, 2, 3, constrain=(WeaveTarget,), one2one=True), APS]


hierarchies = [i[-1] for i in inspect.getmembers(sys.modules[__name__], _predicate)]
