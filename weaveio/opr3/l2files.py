from pathlib import Path
from typing import Union, List

from weaveio.file import File, PrimaryHDU, TableHDU
from weaveio.hierarchy import Multiple, unwind
from weaveio.opr3.hierarchy import ClassificationTable, GalaxyTable, GalaxyL2Spectrum, ClassificationL2Spectrum, APS
from weaveio.opr3.l1files import L1File


class L2File(File):
    match_pattern = '*aps.fit'
    produces = [ClassificationTable, GalaxyTable, ClassificationL2Spectrum, GalaxyL2Spectrum]
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
    def parse_header_runs2source_files(cls, header) -> List[L1File]:
        parent_runids = map(int, header['RUN'].split('+'))
        raise NotImplementedError

    @classmethod
    def read(cls, directory: Union[Path, str], fname: Union[Path, str]) -> 'File':
        fname = Path(fname)
        directory = Path(directory)
        path = directory / fname
        header = cls.read_header(path)
        files = cls.parse_header_runs2source_files(header)
        hdus, file = cls.read_hdus(directory, fname, l1files=files)
        aps = APS(apsvers=header['APSVERS'])
