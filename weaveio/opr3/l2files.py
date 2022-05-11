import sys
from pathlib import Path
from typing import Union, List, Dict, Type, Set, Tuple, Optional

import inspect

from astropy.io import fits
from astropy.io.fits.hdu.base import _BaseHDU
from astropy.table import Table

from weaveio.file import File, PrimaryHDU, TableHDU
from weaveio.graph import Graph
from weaveio.hierarchy import Multiple, unwind, collect, Hierarchy
from weaveio.opr3.hierarchy import APS, FibreTarget, OB, OBSpec, Exposure, WeaveTarget, Fibre, _predicate, RawSpectrum, Run, ArmConfig
from weaveio.opr3.l1 import L1Spectrum, L1SingleSpectrum
from weaveio.opr3.l2 import L2, L2Single, L2OBStack, L2Superstack, L2Supertarget, RedrockIngestedSpectrum, RVSpecFitIngestedSpectrum, FerreIngestedSpectrum, PPXFIngestedSpectrum, GandalfIngestedSpectrum, IngestedSpectrum, Fit, ModelSpectrum, Redrock, RedrockModelSpectrum, \
    RVSpecfitModelSpectrum, RVSpecfit, Ferre, FerreModelSpectrum, PPXFModelSpectrum, PPXF, Gandalf, GandalfModelSpectrum, CombinedIngestedSpectrum, CombinedModelSpectrum, RedrockCombinedIngestedSpectrum, RedrockCombinedModelSpectrum, LogarithmicCombinedModelSpectrum, LogarithmicIngestedSpectrum, \
    LogarithmicCombinedIngestedSpectrum
from weaveio.opr3.l1files import L1File, L1SuperstackFile, L1OBStackFile, L1SingleFile, L1SupertargetFile
from weaveio.writequery import CypherData, groupby
from weaveio.writequery.base import CypherFindReplaceStr


class MissingDataError(Exception):
    pass


def filter_products_from_table(table: Table, maxlength: int) -> Table:
    columns = []
    for i in table.colnames:
        value = table[i]
        if len(value.shape) == 2:
            if value.shape[1] > maxlength:
                continue
        columns.append(i)
    return table[columns]


class L2File(File):
    singular_name = 'l2file'
    is_template = True
    match_pattern = '.*APS.fits'
    antimatch_pattern = '.*cube.*'
    parents = [Multiple(L1File, 2, 3), APS, Multiple(L2)]
    children = []
    hdus = {'primary': PrimaryHDU,
            'class_spectra': TableHDU,
            'galaxy_spectra': TableHDU,
            'stellar_spectra': TableHDU,
            'class_table': TableHDU,
            'stellar_table': TableHDU,
            'galaxy_table': TableHDU}

    @classmethod
    def length(cls, path):
        hdus = fits.open(path)
        names = [i.name for i in hdus]
        return len(hdus[names.index('CLASS_SPECTRA')].data)

    @classmethod
    def decide_filetype(cls, l1filetypes: List[Type[File]]) -> Type[File]:
        l1precedence = [L1SingleFile, L1OBStackFile, L1SuperstackFile, L1SupertargetFile]
        l2precedence = [L2SingleFile, L2OBStackFile, L2SuperstackFile, L2SupertargetFile]
        highest = max(l1precedence.index(l1filetype) for l1filetype in l1filetypes)
        return l2precedence[highest]

    @classmethod
    def match_file(cls, directory: Union[Path, str], fname: Union[Path, str], graph: Graph):
        """
        L2 files can be formed from any combination of L1 files and so the shared hierarchy level can be
        either exposure, OB, OBSpec, or WeaveTarget.
        L2 files are distinguished by the shared hierarchy level of their formative L1 files.
        Therefore, we assign an L2 file to the highest hierarchy level.
        e.g.
        L1Single+L1Single -> L2Single
        L1Stack+L1Single -> L2Stack
        L1SuperStack+L1Stack -> L2SuperStack
        """
        fname = Path(fname)
        directory = Path(directory)
        path = directory / fname
        if not super().match_file(directory, fname, graph):
            return False
        header = cls.read_header_and_aps(path)[0]
        ftypes, _ = zip(*cls.parse_fname(header, fname, instantiate=False))
        return cls.decide_filetype(ftypes) is cls

    @classmethod
    def parser_ftypes_runids(cls, header):
        header_info = [header.get(f'L1_REF_{i}', '.').split('.')[0].split('_') for i in range(4)]
        ftypes_header, runids_header = zip(*[i for i in header_info if len(i) > 1])
        return ftypes_header, list(map(int, runids_header))

    @classmethod
    def parse_fname(cls, header, fname, instantiate=True) -> List[L1File]:
        """
        Return the L1File type and the expected filename that formed this l2 file
        """
        ftype_dict = {
            'single': L1SingleFile,
            'stacked': L1OBStackFile, 'stack': L1OBStackFile,
            'superstack': L1SuperstackFile, 'superstacked': L1SuperstackFile
        }
        split = fname.name.lower().replace('aps.fits', '').replace('aps.fit', '').strip('_.').split('__')
        runids = []
        ftypes = []
        for i in split:
            ftype, runid = i.split('_')
            runids.append(int(runid))
            ftypes.append(str(ftype))
        if len(ftypes) == 1:
            ftypes = [ftypes[0]] * len(runids)  # they all have the same type if there is only one mentioned
        assert len(ftypes) == len(runids), "error parsing runids/types from fname"
        ftypes_header, runids_header = cls.parser_ftypes_runids(header)
        if not all(map(lambda x: x[0] == x[1], zip(runids, runids_header))):
            raise ValueError(f"There is a mismatch between runids in the filename and in in the header")
        if not all(map(lambda x: x[0] == x[1], zip(ftypes, ftypes_header))):
            raise ValueError(f"There is a mismatch between stack/single filetype in the filename and in in the header")
        files = []
        for ftype, runid in zip(ftypes, runids):
            ftype_cls = ftype_dict[ftype]
            fname = ftype_cls.fname_from_runid(runid)
            if instantiate:
                files.append(ftype_cls.find(fname=fname))
            else:
                files.append((ftype_cls, fname))
        return files

    @classmethod
    def find_shared_hierarchy(cls, path: Path) -> Dict:
        raise NotImplementedError

    @classmethod
    def read_header_and_aps(cls, path):
        return fits.open(path)[0].header, fits.open(path)[1].header['APS_V']

    @classmethod
    def read_hdus(cls, directory: Union[Path, str], fname: Union[Path, str], l1files: List[L1File],
                  **hierarchies: Union[Hierarchy, List[Hierarchy]]) -> Tuple[Dict[str, 'HDU'], 'File', List[_BaseHDU]]:
        fdict = {p.plural_name: [] for p in cls.parents if isinstance(p, Multiple) and issubclass(p.node, L1File)} # parse the 1lfile types separately
        for f in l1files:
            fdict[f.plural_name].append(f)
        hierarchies.update(fdict)
        return super().read_hdus(directory, fname, **hierarchies)

    @classmethod
    def read_l2product_table(cls, this_fname, spectrum_hdu, data_hdu, parent_l1filenames,
                             IngestedSpectrumClass: Optional[Type[IngestedSpectrum]],
                             CombinedIngestedSpectrumClass: Optional[Type[CombinedIngestedSpectrum]],
                             FitClass: Optional[Type[Fit]],
                             ModelSpectrumClass: Optional[Type[ModelSpectrum]],
                             CombinedModelSpectrumClass: Optional[Type[CombinedModelSpectrum]],
                             uses_combined_spectrum: Optional[bool],
                             uses_disjoint_spectra: bool,
                             formatter: str,
                             aps):
        if uses_combined_spectrum is None:
            uses_combined_spectrum = any('_C' in i for i in spectrum_hdu.data.names)
        safe_table = filter_products_from_table(Table(data_hdu.data), 10)
        with unwind(CypherData({v: i for i, v in enumerate(spectrum_hdu.data['APS_ID'])}), enumerated=True) as (nrow, fibreid):
            with unwind(CypherData(parent_l1filenames)) as l1_fname:
                l1file = L1File.find(fname=l1_fname)
                l1spectrum = L1Spectrum.find(anonymous_parents=[l1file])
                arm_code = ArmConfig.find(anonymous_children=[l1spectrum])['arm_code']
                if uses_disjoint_spectra:
                    # make an ingested spec for each apsid and arm
                    individual = IngestedSpectrumClass(sourcefile=this_fname, nrow=nrow, name=formatter,
                                                       l1spectra=[l1spectrum], aps=aps)
                    for product in individual.products:
                        column_name = CypherFindReplaceStr(f'{product}_{formatter}_X', arm_code)
                        individual.attach_product(product, spectrum_hdu, nrow, column_name)
                    individual_model = ModelSpectrumClass(sourcefile=this_fname, nrow=nrow, ingested_spectra=individual)
                    for product in individual_model.products:
                        column_name = CypherFindReplaceStr(f'{product}_{formatter}_X', arm_code)
                        individual_model.attach_product(product, spectrum_hdu, nrow, column_name)
            if uses_disjoint_spectra:
                l1files, l1spectra, arm_codes, individuals, individual_models = collect(l1file, l1spectrum, arm_code, individual, individual_model)
                d = {'ingested_spectra': individuals, 'model_spectra': individual_models}
            else:
                l1files, l1spectra, arm_codes = collect(l1file, l1spectrum, arm_code)
                d = {}
            if uses_combined_spectrum:
                combined = CombinedIngestedSpectrumClass(sourcefile=this_fname, nrow=nrow, l1spectra=l1spectra, aps=aps)
                for product in combined.products:
                    column_name = f'{product}_{formatter}'
                    if uses_disjoint_spectra:  # as well, so the colnames differentiate with a `C`
                        column_name += '_C'
                    combined.attach_product(product, spectrum_hdu, nrow, column_name)
                combined_model = ModelSpectrumClass(sourcefile=this_fname, nrow=nrow, ingested_spectra=individual)
                for product in combined_model.products:
                    combined_model.attach_product(product, spectrum_hdu, nrow, f'{product}_{formatter}_C')
                d['combined_ingested_spectrum'] = combined
                d['combined_model_spectrum'] = CombinedModelSpectrumClass(sourcefile=this_fname, nrow=nrow, ingested_spectra=combined)
            fit = FitClass(ingested_spectra=individuals, ingested_combined_spectrum=combined,
                           model_spectra=individual_models, model_combined_spectrum=combined_model,
                           tables=CypherData(safe_table)[nrow])
            return fit


    @classmethod
    def get_l1_filenames(cls, header):
        return [v for k, v in header.items() if 'APS_REF' in k]

    @classmethod
    def read(cls, directory: Union[Path, str], fname: Union[Path, str], slc: slice = None):
        fname = Path(fname)
        directory = Path(directory)
        path = directory / fname
        header, aps = cls.read_header_and_aps(path)
        # find L1 files in database and use them to instantiate a new L2 file
        l1files = cls.parse_fname(header, fname)
        aps = APS(version=aps)
        hierarchies = cls.find_shared_hierarchy(path)
        astropy_hdus = fits.open(path)
        fnames = cls.get_l1_filenames(header)
        cls.read_l2product_table(path, astropy_hdus[4], astropy_hdus[1], fnames,
                                 RedrockIngestedSpectrum, RedrockCombinedIngestedSpectrum, Redrock,
                                 RedrockModelSpectrum, RedrockCombinedModelSpectrum,
                                 None, True, 'RR', aps)
        cls.read_l2product_table(path, astropy_hdus[5], astropy_hdus[2], fnames,
                                 IngestedSpectrum, CombinedIngestedSpectrum, RVSpecfit,
                                 ModelSpectrum, CombinedModelSpectrum,
                                 None, True, 'RVS', aps)
        cls.read_l2product_table(path, astropy_hdus[5], astropy_hdus[2], fnames,
                                 IngestedSpectrum, CombinedIngestedSpectrum, Ferre,
                                 ModelSpectrum, CombinedModelSpectrum,
                                 None, True, 'FR', aps)
        cls.read_l2product_table(path, astropy_hdus[6], astropy_hdus[3], fnames,
                                 None, LogarithmicCombinedIngestedSpectrum, PPXF,
                                 None, LogarithmicCombinedModelSpectrum,
                                 True, False, 'PPXF', aps)
        cls.read_l2product_table(path, astropy_hdus[6], astropy_hdus[3], fnames,
                                 None, GandalfIngestedSpectrum, Gandalf,
                                 None, GandalfModelSpectrum,
                                 True, False, 'GAND', aps)
        hdu_nodes, file, astropy_hdus = cls.read_hdus(directory, fname, l1files=l1files, aps=aps, **hierarchies)


class L2SingleFile(L2File):
    singular_name = 'l2single_file'
    children = []
    parents = [Multiple(L1SingleFile, 2, 3, constrain=(Exposure,)), APS, Multiple(L2Single)]


    @classmethod
    def find_shared_hierarchy(cls, path) -> Dict:
        header = cls.read_header_and_aps(path)[0]
        runids = cls.parser_ftypes_runids(header)[1]
        run = Run.find(id=runids[0])
        return {'exposure': Exposure.find(anonymous_children=[run])}


class L2OBStackFile(L2File):
    singular_name = 'l2obstack_file'
    children = []
    parents = [Multiple(L1OBStackFile, 1, 3, constrain=(OB,)), APS, Multiple(L2OBStack)]

    @classmethod
    def find_shared_hierarchy(cls, path) -> Dict:
        header = cls.read_header_and_aps(path)[0]
        return {'ob': OB.find(obid=header['OBID'])}


class L2SuperstackFile(L2File):
    singular_name = 'l2superstack_file'
    children = []
    parents = [Multiple(L1SingleFile, 0, 3, constrain=(OBSpec,)),
               Multiple(L1OBStackFile, 0, 3, constrain=(OBSpec,)),
               Multiple(L1SuperstackFile, 0, 3, constrain=(OBSpec,)), APS,
               Multiple(L2Superstack)]

    @classmethod
    def find_shared_hierarchy(cls, path) -> Dict:
        header = cls.read_header_and_aps(path)[0]
        return {'obspec': OBSpec.find(xml=str(header['cat-name']))}


class L2SupertargetFile(L2File):
    singular_name = 'l2supertarget_file'
    match_pattern = 'WVE_*aps.fits'
    children = []
    parents = [Multiple(L1SupertargetFile, 2, 3, constrain=(WeaveTarget,)), APS, L2Supertarget]

    @classmethod
    def parse_fname(cls, header, fname, instantiate=True) -> List[L1File]:
        raise NotImplementedError

    @classmethod
    def find_shared_hierarchy(cls, path: Path) -> Dict:
        hdus = fits.open(path)
        names = [i.name for i in hdus]
        cname = hdus[names.index('CLASS_TABLE')].data['CNAME'][0]
        return {'weavetarget': WeaveTarget.find(cname=cname)}


hierarchies = [i[-1] for i in inspect.getmembers(sys.modules[__name__], _predicate)]