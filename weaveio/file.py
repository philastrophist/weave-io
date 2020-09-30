from functools import lru_cache
from pathlib import Path
from typing import Union, Tuple

import pandas as pd
from astropy.io import fits
from astropy.table import Table as AstropyTable

from weaveio.config_tables import progtemp_config
from weaveio.graph import Graph
from weaveio.hierarchy import Run, OBRealisation, OBSpec, ArmConfig, Target, Exposure, Multiple, TargetSet, ProgTemp, ObsTemp, Graphable, l1file_attrs
from weaveio.product import Header, Array, Table


class File(Graphable):
    idname = 'fname'
    constructed_from = []
    indexable_by = []
    products = {}
    type_graph_attrs = l1file_attrs
    concatenation_constant_names = {}

    def __init__(self, fname: Union[Path, str], **kwargs):
        self.index = None
        self.fname = Path(fname)
        self.identifier = str(self.fname)
        self.name = f'{self.__class__.__name__}({self.fname})'
        if len(kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)
            self.predecessors = kwargs
        else:
            self.predecessors = self.read()
        super(File, self).__init__(**self.predecessors)
        self.product_data = {}

    def match(self, directory: Path):
        raise NotImplementedError

    @property
    def graph_name(self):
        return f"File({self.fname})"

    def read(self):
        raise NotImplementedError

    def build_index(self) -> None:
        if self.index is not None:
            self.index['rowid'] = range(len(self.index))
            self.index['fname'] = self.fname

    def match_index(self, index) -> pd.DataFrame:
        self.build_index()
        keys = [i for i in index.columns if i not in ['fname', 'rowid']]
        for i, k in enumerate(keys):
            exists = index[k].isin(self.index[k])
            if not exists.all():
                raise KeyError(f"{index[~exists].values} are not found")
            f = self.index[k].isin(index[k])
            if i == 0:
                filt = f
            else:
                filt &= f
        return filt

    def read_concatenation_constants(self, product_name) -> Tuple:
        raise NotImplementedError

    def is_concatenatable_with(self, other: 'File', product_name) -> bool:
        if self.products[product_name] is not other.products[product_name]:
            return False
        self_values = self.read_concatenation_constants(product_name)
        other_values = other.read_concatenation_constants(product_name)
        return self_values == other_values

    def read_product(self, product_name):
        self.build_index()
        return getattr(self, f'read_{product_name}')()


class HeaderFibinfoFile(File):
    fibinfo_i = -1
    spectral_concatenation_constants = ['CRVAL1', 'CD1_1', 'NAXIS1']
    products = {'primary': Header, 'data': Array, 'ivar': Array, 'data_noss': Array,
                'ivar_noss': Array, 'sensfunc': Array, 'fibtable': Table}
    concatenation_constant_names = {'primary': True, 'data': spectral_concatenation_constants,
                               'ivar': spectral_concatenation_constants,
                               'data_noss': spectral_concatenation_constants,
                               'ivar_noss': spectral_concatenation_constants,
                               'sens_func': spectral_concatenation_constants,
                               'fibtable': ['NAXIS1']}
    hdus = ['primary', 'data', 'ivar', 'data_noss', 'ivar_noss', 'sensfunc', 'fibtable']

    def read_concatenation_constants(self, product_name) -> Tuple:
        header = fits.open(self.fname)[self.hdus.index(product_name)].header
        return tuple(header[c] for c in self.concatenation_constant_names[product_name])

    def read(self):
        header = fits.open(self.fname)[0].header
        runid = str(header['RUN'])
        camera = str(header['CAMERA'].lower()[len('WEAVE'):])
        expmjd = str(header['MJD-OBS'])
        res = str(header['VPH']).rstrip('123')
        obstart = str(header['OBSTART'])
        obtitle = str(header['OBTITLE'])
        obid = str(header['OBID'])

        fibinfo = self._read_fibtable()
        progtemp = ProgTemp.from_progtemp_code(header['PROGTEMP'])
        vph = int(progtemp_config[(progtemp_config['mode'] == progtemp.mode)
                              & (progtemp_config['resolution'] == res)][f'{camera}_vph'].iloc[0])
        armconfig = ArmConfig(vph=vph, resolution=res, camera=camera)  # must instantiate even if not used
        obstemp = ObsTemp.from_header(header)
        targetset = TargetSet.from_fibinfo(fibinfo)
        obspec = OBSpec(targetset=targetset, obtitle=obtitle, obstemp=obstemp, progtemp=progtemp)
        obrealisation = OBRealisation(obid=obid, obstartmjd=obstart, obspec=obspec)
        exposure = Exposure(expmjd=expmjd, obrealisation=obrealisation)
        run = Run(runid=runid, camera=camera, exposure=exposure)
        return {'run': [run]}

    def build_index(self) -> None:
        if self.index is None:
            self.index = pd.DataFrame({'cname': [i.cname for i in self.targets]})
        super(HeaderFibinfoFile, self).build_index()

    def read_primary(self):
         return Header(fits.open(self.fname)[0].header, self.index)

    def read_data(self):
        return Array(fits.open(self.fname)[1].data, self.index)

    def read_ivar(self):
        return Array(fits.open(self.fname)[2].data, self.index)

    def read_data_noss(self):
        return Array(fits.open(self.fname)[3].data, self.index)

    def read_ivar_noss(self):
        return Array(fits.open(self.fname)[4].data, self.index)

    def read_sens_func(self):
        return Array(fits.open(self.fname)[5].data, self.index)

    def _read_fibtable(self):
        return AstropyTable(fits.open(self.fname)[self.fibinfo_i].data)[:5]

    def read_fibtable(self):
        return Table(self._read_fibtable(), self.index)



class Raw(HeaderFibinfoFile):
    parents = [Run]
    fibinfo_i = 3

    @classmethod
    def match(cls, directory: Path):
        return directory.glob('r*.fit')


class L1Single(HeaderFibinfoFile):
    parents = [Run]
    constructed_from = [Raw]

    @classmethod
    def match(cls, directory):
        return directory.glob('single_*.fit')


class L1Stack(HeaderFibinfoFile):
    parents = [OBRealisation]
    factors = ['VPH']
    constructed_from = [L1Single]

    @classmethod
    def match(cls, directory):
        return directory.glob('stacked_*.fit')


class L1SuperStack(File):
    parents = [OBSpec]
    factors = ['VPH']
    constructed_from = [L1Single]

    @classmethod
    def match(cls, directory):
        return directory.glob('superstacked_*.fit')


class L1SuperTarget(File):
    parents = [ArmConfig, Target]
    factors = ['Binning', 'Mode']
    constructed_from = [L1Single]

    @classmethod
    def match(cls, directory):
        return directory.glob('[Lm]?WVE_*.fit')


class L2Single(File):
    parents = [Exposure]
    constructed_from = [Multiple(L1Single, 2, 2)]

    @classmethod
    def match(cls, directory):
        return directory.glob('single_*_aps.fit')


class L2Stack(File):
    parents = [Multiple(ArmConfig, 1, 3), TargetSet]
    factors = ['Binning', 'Mode']
    constructed_from = [Multiple(L1Stack, 0, 3), Multiple(L1SuperStack, 0, 3)]

    @classmethod
    def match(cls, directory):
        return directory.glob('(super)?stacked_*_aps.fit')


class L2SuperTarget(File):
    parents = [Multiple(ArmConfig, 1, 3), Target]
    factors = ['Mode', 'Binning']
    constructed_from = [Multiple(L1SuperTarget, 2, 3)]

    @classmethod
    def match(cls, directory):
        return directory.glob('[Lm]?WVE_*_aps.fit')
