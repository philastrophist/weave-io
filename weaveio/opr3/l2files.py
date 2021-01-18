from pathlib import Path
from typing import Union, List, Dict, Type, Set

from astropy.io import fits
from astropy.table import Table

from weaveio.file import File, PrimaryHDU, TableHDU
from weaveio.graph import Graph
from weaveio.hierarchy import Multiple, unwind, collect
from weaveio.opr3.hierarchy import APS, L1SpectrumRow, FibreTarget, OB, OBSpec, L2Stack, L2SuperStack, L2SuperTarget, L2Single, ClassificationTable, GalaxyTable, Exposure, WeaveTarget, L2
from weaveio.opr3.l1files import L1File, L1SuperStackFile, L1StackFile, L1SingleFile, L1SuperTargetFile
from weaveio.writequery import CypherData, groupby


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
    is_template = True
    match_pattern = '*aps.fits'
    produces = [L2]
    corresponding_hdus = {'class_table': ClassificationTable, 'galaxy_table': GalaxyTable}#, 'class_spectra', 'galaxy_spectra']
    parents = [Multiple(L1File, 2, 3), APS]
    hdus = {'primary': PrimaryHDU, 'fibtable': TableHDU,
            'class_spectra': TableHDU,
            'stellar_spectra_ferre': TableHDU, 'stellar_spectra_rvs': TableHDU,
            'galaxy_spectra': TableHDU,
            'class_table': TableHDU,
            'stellar_table': TableHDU,
            'stellar_table_rvs': TableHDU,
            'galaxy_table': TableHDU}
    recommended_batchsize = 600

    @classmethod
    def length(cls, path):
        hdus = fits.open(path)
        names = [i.name for i in hdus]
        return len(hdus[names.index('CLASS_SPECTRA')].data)

    @classmethod
    def decide_filetype(cls, l1filetypes: List[Type[File]]) -> Type[File]:
        l1precedence = [L1SingleFile, L1StackFile, L1SuperStackFile, L1SuperTargetFile]
        l2precedence = [L2SingleFile, L2StackFile, L2SuperStackFile, L2SuperTargetFile]
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
            'stacked': L1StackFile, 'stack': L1StackFile,
            'superstack': L1SuperStackFile, 'superstacked': L1SuperStackFile
        }
        split = fname.name.replace('.aps.fits', '').replace('.aps.fit', '').split('_')
        runids = []
        ftypes = []
        for i in split:
            try:
                runids.append(int(i))
            except ValueError:
                ftypes.append(str(i))
        if len(ftypes) == 1:
            ftypes = [ftypes[0]] * len(runids)  # they all have the same type if there is only one mentioned
        assert len(ftypes) == len(runids), "error parsing runids/types from fname"
        assert all(int(i) in runids for i in header['RUN'].split('+')), "fname runids and header runids do not match"
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
    def get_all_nspecs(cls, astropy_hdus) -> List[int]:
        nspecs = set()
        for hdu in astropy_hdus:
            try:
                nspecs.update(hdu.data['nspec'].tolist())
            except (TypeError, ):  # dont do the primary header
                pass
        return list(nspecs)


    @classmethod
    def read(cls, directory: Union[Path, str], fname: Union[Path, str], slc: slice = None):
        fname = Path(fname)
        directory = Path(directory)
        path = directory / fname
        header = cls.read_header(path)
        l1files = cls.parse_fname(header, fname)
        aps = APS(apsvers=header['APSVERS'])
        hierarchies = cls.find_shared_hierarchy(path)
        hdu_nodes, file, astropy_hdus = cls.read_hdus(directory, fname, l1files=l1files, aps=aps, **hierarchies)
        produce = cls.produces[0]
        l1filenames = CypherData([l.fname for l in l1files], 'l1fnames')
        nspecs = CypherData(cls.get_all_nspecs(astropy_hdus), 'nspecs')
        with unwind(nspecs) as nspec:
            with unwind(l1filenames) as l1fname:
                spectrum = L1SpectrumRow.find(sourcefile=l1fname, nrow=nspec)
            l1spectra = collect(spectrum)
        #     if not issubclass(cls, L2SuperTargetFile):
        #         fibretarget = FibreTarget.find(anonymous_children=[l1spectra[0]])
        #         hierarchies['fibretarget'] = fibretarget
        #     l2 = produce(sourcefile=file.fname, nrow=nspec, l1spectrumrows=l1spectra, aps=aps, **hierarchies)
        # l2s = collect(l2)
        # l2_dict = groupby(l2s, 'nrow')
        # for name in cls.corresponding_hdus.keys():
        #     cls.make_data_rows(name, slc, file, l2_dict, astropy_hdus, hdu_nodes)
        # return file

    @classmethod
    def read_one_hdu_l2data(cls, hdus, hduname, slc=None):
        slc = slice(None) if slc is None else slc
        names = [i.name.lower().strip() for i in hdus]
        table = Table(hdus[names.index(hduname)].data)[slc]
        if len(table.colnames):
            table.rename_columns(table.colnames, [i.lower() for i in table.colnames])
            table['spec_index'] = range(len(table))
        table = filter_products_from_table(table, 10)  # removes huge arrays that are best kept in binary files
        data = CypherData(table, hduname)
        return data

    @classmethod
    def make_data_rows(cls, hduname, slc, file, l2_by_nrow, astropyhdulist, hdus):
        """
        For each L2 row, there are 2-3 L1 rows
        Since we know the L1 files, we can identify those rows with nspec since they are the same
        """
        row_type = cls.corresponding_hdus[hduname]
        table = cls.read_one_hdu_l2data(astropyhdulist, hduname, slc)
        with unwind(table) as row:
            nspec = row['nspec']
            l2row = row_type(sourcefile=file.fname, nrow=nspec, l2=l2_by_nrow[nspec], tables=row)
            l2row.attach_products(file, index=row['spec_index'], **hdus)
        l2rows = collect(l2row)
        return l2rows


class L2SingleFile(L2File):
    produces = [L2Single]
    parents = [Multiple(L1File, 2, 3), Exposure, APS]

    @classmethod
    def find_shared_hierarchy(cls, path) -> Dict:
        header = cls.read_header(path)
        return {'exposure': Exposure.find(obid=header['MJD-OBS'])}


class L2StackFile(L2File):
    produces = [L2Stack]
    parents = [Multiple(L1File, 2, 3), OB, APS]

    @classmethod
    def find_shared_hierarchy(cls, path) -> Dict:
        header = cls.read_header(path)
        return {'ob': OB.find(obid=header['OBID'])}


class L2SuperStackFile(L2File):
    produces = [L2SuperStack]
    parents = [Multiple(L1File, 2, 3), OBSpec, APS]

    @classmethod
    def find_shared_hierarchy(cls, path) -> Dict:
        header = cls.read_header(path)
        return {'obspec': OBSpec.find(xml=str(header['cat-name']))}


class L2SuperTargetFile(L2File):
    match_pattern = 'WVE_*aps.fits'
    produces = [L2SuperTarget]
    parents = [Multiple(L1File, 2, 3), WeaveTarget, APS]

    @classmethod
    def parse_fname(cls, header, fname, instantiate=True) -> List[L1File]:
        raise NotImplementedError

    @classmethod
    def find_shared_hierarchy(cls, path: Path) -> Dict:
        hdus = fits.open(path)
        names = [i.name for i in hdus]
        cname = hdus[names.index('CLASS_TABLE')].data['CNAME'][0]
        return {'weavetarget': WeaveTarget.find(cname=cname)}
