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
    RVSpecfitModelSpectrum, RVSpecfitVersion, RVSpecfit, Ferre, FerreModelSpectrum, FerreVersion, PPXFModelSpectrum, PPXF, PPXFVersion, Gandalf, GandalfModelSpectrum, GandalfVersion, RedrockVersion
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
    software_versions = [RedrockVersion, PPXFVersion, GandalfVersion, FerreVersion, RVSpecfitVersion]
    parents = [Multiple(L1File, 2, 3), APS, Multiple(L2)]
    children = []
    produces = software_versions
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
        header = cls.read_header(path)
        ftypes, _ = zip(*cls.parse_fname(header, fname, instantiate=False))
        return cls.decide_filetype(ftypes) is cls

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
        header_info = [header.get(f'L1_REF_{i}', '.').split('.')[0].split('_') for i in range(4)]
        ftypes_header, runids_header = zip(*[i for i in header_info if len(i) > 1])
        runids_header = list(map(int, runids_header))
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
    def read_header(cls, path):
        return fits.open(path)[0].header

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
                             IngestedSpectrumClass: Type[IngestedSpectrum],
                             FitClass: Type[Fit],
                             ModelSpectrumClass: Type[ModelSpectrum],
                             fit_version,
                             uses_combined_spectrum: Optional[bool],
                             uses_disjoint_spectra: bool,
                             formatter: str,
                             aps):
        if uses_combined_spectrum is None:
            uses_combined_spectrum = any('_C' in i for i in spectrum_hdu.data.names)
        safe_table = filter_products_from_table(Table(data_hdu.data), 10)
        aps_ids_nrows = CypherData({v: i for i, v in enumerate(spectrum_hdu.data['APS_ID'])})

        l1files = [L1File.find(fname=fname) for fname in parent_l1filenames]  # for each l1 file that makes this L2
        l1spectra = [L1Spectrum.find(anonymous_parents=[l1file]) for l1file in l1files]
        singles = [L1SingleSpectrum.find(anonymous_parents=[l1spectrum]) for l1spectrum in l1spectra]
        raws = [RawSpectrum.find(anonymous_parents=[single]) for single in singles]
        runs = [Run.find(anonymous_parents=[raw]) for raw in raws]
        arms = [ArmConfig.find(anonymous_parents=[run])['colour_code'] for run in runs]
        fibretargets = [FibreTarget.find(anonymous_children=[l1spectrum]) for l1spectrum in l1spectra]
        fibreids = [Fibre.find(anonymous_children=[fibretarget])['id'] for fibretarget in fibretargets]
        nrow = aps_ids_nrows[fibreids[0]]
        # we are now at the level of l1spectra/divided by file in python
        ingested_spectra = []
        if uses_disjoint_spectra:
            # make an ingested spec for each apsid and arm
            for arm, l1spectrum in zip(arms, l1spectra):
                individual = IngestedSpectrumClass(sourcefile=this_fname, nrow=nrow,
                                                   l1spectra=l1spectrum, aps=aps, arm=arm)
                for product in individual.products:
                    column_name = CypherFindReplaceStr(f'{product}_{formatter}_X', arm)
                    individual.attach_product(product, spectrum_hdu, nrow, column_name)

                ingested_spectra.append(individual)
        if uses_combined_spectrum:
            combined = IngestedSpectrumClass(sourcefile=this_fname, nrow=aps_ids_nrows[fibreids[0]],
                                             l1spectra=l1spectra, aps=aps)
            for product in combined.products:
                column_name = f'{product}_{formatter}'
                if IngestedSpectrumClass not in [GandalfIngestedSpectrum, PPXFIngestedSpectrum]:
                    column_name += '_C'
                combined.attach_product(product, spectrum_hdu, nrow, column_name)
            ingested_spectra.append(combined)
        fit = FitClass(**{fit_version.singular_name: fit_version,
                          IngestedSpectrumClass.plural_name: ingested_spectra},
                 tables=CypherData(safe_table)[aps_ids_nrows[fibreids[0]]])
        for arm, ingested_spectrum in zip(arms, ingested_spectra):
            individual_model = ModelSpectrumClass(sourcefile=this_fname,
                                                  nrow=aps_ids_nrows[fibreids[0]],
                                                  **{FitClass.singular_name:fit,
                                                     IngestedSpectrumClass.plural_name: ingested_spectrum})
            for product in individual_model.products:
                column_name = CypherFindReplaceStr(f'{product}_{formatter}_X', arm)
                individual_model.attach_product(product, spectrum_hdu, nrow, column_name)

    @classmethod
    def get_l1_filenames(cls, header):
        return [v for k, v in header.items() if 'APS_REF' in k]

    @classmethod
    def read(cls, directory: Union[Path, str], fname: Union[Path, str], slc: slice = None):
        fname = Path(fname)
        directory = Path(directory)
        path = directory / fname
        header = cls.read_header(path)
        # find L1 files in database and use them to instantiate a new L2 file
        l1files = cls.parse_fname(header, fname)
        aps = APS(apsvers=header['APSVERS'])
        hierarchies = cls.find_shared_hierarchy(path)
        hdu_nodes, file, astropy_hdus = cls.read_hdus(directory, fname, l1files=l1files, aps=aps, **hierarchies)
        fnames = cls.get_l1_filenames(header)
        cls.read_l2product_table(path, astropy_hdus[4], astropy_hdus[1], fnames,
                                 RedrockIngestedSpectrum, Redrock,
                                 RedrockModelSpectrum, RedrockVersion.from_header(header),
                                 None, True, 'RR', aps)
        cls.read_l2product_table(path, astropy_hdus[5], astropy_hdus[2], fnames,
                                 RVSpecFitIngestedSpectrum, RVSpecfit,
                                 RVSpecfitModelSpectrum, RVSpecfitVersion.from_header(header),
                                 None, True, 'RVS', aps)
        cls.read_l2product_table(path, astropy_hdus[5], astropy_hdus[2], fnames,
                                 FerreIngestedSpectrum, Ferre,
                                 FerreModelSpectrum, FerreVersion.from_header(header),
                                 None, True, 'FR', aps)
        cls.read_l2product_table(path, astropy_hdus[6], astropy_hdus[3], fnames,
                                 PPXFIngestedSpectrum, PPXF,
                                 PPXFModelSpectrum, PPXFVersion.from_header(header),
                                 True, False, 'PPXF', aps)
        cls.read_l2product_table(path, astropy_hdus[6], astropy_hdus[3], fnames,
                                 GandalfIngestedSpectrum, Gandalf,
                                 GandalfModelSpectrum, GandalfVersion.from_header(header),
                                 True, False, 'GAND', aps)


class L2SingleFile(L2File):
    singular_name = 'l2single_file'
    children = []
    parents = [Multiple(L1SingleFile, 2, 3, constrain=(Exposure,)), APS, Multiple(L2Single)]


    @classmethod
    def find_shared_hierarchy(cls, path) -> Dict:
        header = cls.read_header(path)
        return {'exposure': Exposure.find(obid=header['MJD-OBS'])}


class L2OBStackFile(L2File):
    singular_name = 'l2obstack_file'
    children = []
    parents = [Multiple(L1OBStackFile, 1, 3, constrain=(OB,)), APS, Multiple(L2OBStack)]

    @classmethod
    def find_shared_hierarchy(cls, path) -> Dict:
        header = cls.read_header(path)
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
        header = cls.read_header(path)
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