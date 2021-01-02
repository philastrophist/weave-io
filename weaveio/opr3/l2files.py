from pathlib import Path
from typing import Union, List, Dict

from astropy.io import fits
from astropy.table import Table

from weaveio.file import File, PrimaryHDU, TableHDU
from weaveio.graph import Graph
from weaveio.hierarchy import Multiple, unwind, collect
from weaveio.opr3.hierarchy import ClassificationTable, GalaxyTable, GalaxySpectrum, ClassificationSpectrum, APS, L1SpectrumRow, FibreTarget, L2_FTYPES, L2_DTYPES, OB, OBSpec
from weaveio.opr3.l1files import L1File, L1SuperStackFile, L1StackFile
from weaveio.writequery import CypherData


# make new types based on stack level and type of data
L2_TYPES = {}
for ftype in L2_FTYPES:
    for dtype in L2_DTYPES:
        name = ftype.__name__.replace('L2', '') + dtype.__name__
        L2_TYPES[name] = type(name, (ftype, dtype), {'is_template': False})


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
    match_pattern = '*aps.fit'
    produces = [ClassificationTable, GalaxyTable]#, ClassificationSpectrum, GalaxySpectrum]
    corresponding_hdus = ['class_table', 'galaxy_table']#, 'class_spectra', 'galaxy_spectra']
    parents = [Multiple(L1File, 2, 3)]
    hdus = {'primary': PrimaryHDU, 'fibtable': TableHDU,
            'class_spectra': TableHDU,
            'stellar_spectra_ferre': TableHDU, 'stellar_spectra_rvs': TableHDU,
            'galaxy_spectra': TableHDU,
            'class_table': TableHDU,
            'stellar_table': TableHDU,
            'stellar_table_rvs': TableHDU,
            'galaxy_table': TableHDU}

    @classmethod
    def query_structure(cls, path, graph):
        header = cls.read_header(path)
        runids = map(int, header['RUN'].split('+'))
        result = graph.execute('UNWIND $runids as runid MATCH (run:Run {runid:runid}) '
                               'MATCH (run)<-[*]-(ob:OB)'
                               'MATCH (ob)<--(obspec:OBSpec) '
                               'return collect(distinct ob.obid) as obids, collect(distinct obspec.xml) as xmls',
                               runids=runids).to_table()

        if not len(result):
            raise MissingDataError(f"Data for runs {runids} has not been found in the database. "
                                   f"Unable to decide what type of file {path} is.")
        obids, xmls = result[0][0], result[0][1]
        return obids, xmls

    @classmethod
    def parse_fname(cls, header, fname, instantiate=True) -> List[L1File]:
        ftype_dict = {'stacked': L1StackFile, 'stack': L1StackFile,
                      'superstack': L1SuperStackFile, 'superstacked': L1SuperStackFile}
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
    def read(cls, directory: Union[Path, str], fname: Union[Path, str]):
        fname = Path(fname)
        directory = Path(directory)
        path = directory / fname
        header = cls.read_header(path)
        l1files = cls.parse_fname(header, fname)
        hdus, file = cls.read_hdus(directory, fname, l1files=l1files)
        astropyhdulist = fits.open(path)
        aps = APS(apsvers=header['APSVERS'])
        hierarchies = cls.find_shared_hierarchy(path)
        for name in cls.corresponding_hdus:
            cls.make_data_rows(name, l1files, file, astropyhdulist, hdus, aps, **hierarchies)
        return file

    @classmethod
    def read_one_hdu_l2data(cls, fname, hdus, hduname, l1files):
        names = [i.name.lower().strip() for i in hdus]
        table = Table(hdus[names.index(hduname)].data)
        if len(table.colnames):
            table.rename_columns(table.colnames, [i.lower() for i in table.colnames])
            table['spec_index'] = range(len(table))
        table = filter_products_from_table(table, 10)  # removes huge arrays that are best kept in binary files
        hashids = []
        for row in table:
            nspec = row['nspec']
            hashids.append([f'{l1file.fname}[{nspec}]' for l1file in l1files])
        table['spectrum_hashids'] = hashids
        table['hashid'] = [f'{fname}[{row["nspec"]}]' for row in table]
        data = CypherData(table, hduname)
        return data

    @classmethod
    def make_data_rows(cls, hduname, l1files, file, astropyhdulist, hdus, aps, **hierarchies):
        row_type = cls.produces[cls.corresponding_hdus.index(hduname)]
        table = cls.read_one_hdu_l2data(file.fname, astropyhdulist, hduname, l1files)
        with unwind(table) as row:
            with unwind(row['spectrum_hashids']) as hashid:
                spectrum = L1SpectrumRow.find(hashid=hashid)
            spectra = collect(spectrum)
            spectra_list = [spectra[i] for i, _ in enumerate(l1files)]
            fibretarget = FibreTarget.find(anonymous_children=[spectra[0]])
            l2row = row_type(hashid=table['hashid'], tables=row, l1spectrumrows=spectra_list, fibretarget=fibretarget, aps=aps, **hierarchies)
            l2row.attach_products(file, index=row['spec_index'], **hdus)
        l2rows = collect(l2row)
        return l2rows


class StackL2File(L2File):
    produces = [L2_TYPES['StackClassificationTable'], L2_TYPES['StackGalaxyTable']]#,
                #L2_TYPES['StackClassificationSpectrum'], L2_TYPES['StackGalaxySpectrum']]
    corresponding_hdus = ['class_table', 'galaxy_table',]# 'class_spectra', 'galaxy_spectra']

    @classmethod
    def match_file(cls, directory: Union[Path, str], fname: Union[Path, str], graph: Graph):
        fname = Path(fname)
        directory = Path(directory)
        path = directory / fname
        if not super().match_file(directory, fname, graph):
            return False
        obids, xmls = cls.query_structure(path, graph)
        if len(obids) == 1:
            return True
        return False

    @classmethod
    def find_shared_hierarchy(cls, path) -> Dict:
        header = cls.read_header(path)
        return {'ob': OB.find(obid=int(header['OBID']))}


class SuperStackL2File(L2File):
    produces = [L2_TYPES['SuperStackClassificationTable'], L2_TYPES['SuperStackGalaxyTable']]#,
                #L2_TYPES['SuperStackClassificationSpectrum'], L2_TYPES['SuperStackGalaxySpectrum']]
    corresponding_hdus = ['class_table', 'galaxy_table',]# 'class_spectra', 'galaxy_spectra']

    @classmethod
    def match_file(cls, directory: Union[Path, str], fname: Union[Path, str], graph: Graph):
        fname = Path(fname)
        directory = Path(directory)
        path = directory / fname
        if not super().match_file(directory, fname, graph):
            return False
        obids, xmls = cls.query_structure(path, graph)
        if len(obids) == 1:
            return False
        elif len(xmls) == 1:
            return True
        else:
            return False

    @classmethod
    def find_shared_hierarchy(cls, path) -> Dict:
        header = cls.read_header(path)
        return {'obspec': OBSpec.find(xml=str(header['cat-name']))}
