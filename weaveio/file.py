from collections import defaultdict
from functools import lru_cache
from pathlib import Path
from typing import Union, Tuple, Dict

import pandas as pd
from astropy.io import fits
from astropy.table import Table as AstropyTable

from weaveio.config_tables import progtemp_config
from weaveio.graph import Graph, Unwind, ContextError
from weaveio.hierarchy import Run, OBRealisation, OBSpec, ArmConfig, Exposure, \
    Multiple, FibreSet, ProgTemp, ObsTemp, Graphable, l1file_attrs, Survey, Fibre, FibreAssignment, WeaveTarget, OBSpecTarget, FibreTarget
from weaveio.product import Header, Array, Table


class File(Graphable):
    idname = 'fname'
    constructed_from = []
    indexable_by = []
    products = {}
    factors = []
    type_graph_attrs = l1file_attrs
    concatenation_constant_names = {}

    def __repr__(self):
        return f'<{self.singular_name}(fname={self.fname})>'

    def __init__(self, fname: Union[Path, str], **kwargs):
        self.index = None
        self.fname = Path(fname)
        self.identifier = str(self.fname)
        self.name = f'{self.__class__.__name__}({self.fname})'
        if len(kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)
            predecessors = kwargs
        else:
            try:
                predecessors, factors = self.read()
            except ContextError:
                return
        self.predecessors = {}
        fs = {f: '...' for f in self.factors}
        exception = TypeError(f"{self}.read() must return dict of {self.parents}, {fs}")
        for p in self.parents:
            if isinstance(p, Multiple):
                thing = predecessors[p.plural_name]
                if not isinstance(thing, (list, tuple)):
                    raise exception
                if not all(isinstance(pred, p.node) for pred in thing):
                    raise exception
                self.predecessors[p.plural_name] = thing
            else:
                thing = predecessors[p.singular_name]
                if isinstance(thing, Unwind):
                    self.predecessors[p.plural_name] = thing
                else:
                    self.predecessors[p.singular_name] = [thing]
        for f in self.factors:
            setattr(self, f, factors[f])
        for p, v in self.predecessors.items():
            setattr(self, p, v[0])
        super(File, self).__init__(**self.predecessors)
        self.product_data = {}

    @classmethod
    def match(cls, directory: Path):
        raise NotImplementedError

    @property
    def graph_name(self):
        return f"File({self.fname})"

    def read(self) -> Tuple[Dict[str, 'Hierarchy']]:
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
    products = {'primary': Header, 'fluxes': Array, 'ivars': Array, 'fluxes_noss': Array,
                'ivars_noss': Array, 'sensfuncs': Array, 'fibtable': Table}
    concatenation_constant_names = {'primary': True, 'fluxes': spectral_concatenation_constants,
                               'ivars': spectral_concatenation_constants,
                               'fluxes_noss': spectral_concatenation_constants,
                               'ivars_noss': spectral_concatenation_constants,
                               'sens_funcs': spectral_concatenation_constants,
                               'fibtable': ['NAXIS1']}
    product_indexables = {'primary': None, 'fluxes': 'cname',
                          'ivars':  'cname', 'fluxes_noss':  'cname',
                          'ivars_noss':  'cname',
                          'sensfuncs':  'cname', 'fibtable':  'cname'}
    hdus = ['primary', 'fluxes', 'ivars', 'fluxes_noss', 'ivars_noss', 'sensfuncs', 'fibtable']

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
        catname = str(header['CAT-NAME'])
        obid = str(header['OBID'])

        progtemp = ProgTemp.from_progtemp_code(header['PROGTEMP'])
        vph = int(progtemp_config[(progtemp_config['mode'] == progtemp.instrumentconfiguration.mode)
                                  & (progtemp_config['resolution'] == res)][f'{camera}_vph'].iloc[0])
        arm = ArmConfig(vph=vph, resolution=res, camera=camera)  # must instantiate even if not used
        obstemp = ObsTemp.from_header(header)

        graph = Graph.get_context()
        fibinfo = self._read_fibtable().to_pandas()
        fibinfo['TARGSRVY'] = fibinfo['TARGSRVY'].str.replace(' ', '')
        with graph.add_table(fibinfo, index='fibreid', split=[('targsrvy', ',')]) as table:
            surveys = Survey(surveyname=table['targsrvy'])
            weavetargets = WeaveTarget(cname=table['cname'], surveys=surveys)
            fibres = Fibre(fibreid=table['fibreid'])
            fibreassignments = FibreAssignment(weavetarget=weavetargets, fibre=fibres, tables=table)
            obspectargets = OBSpecTarget(obid=obid, tables=table)
            fibretarget = FibreTarget(fibreassignment=fibreassignments, obspectarget=obspectargets)
            fibreset = FibreSet(fibretargets=fibretarget, obid=obid)
        obspec = OBSpec(catname=catname, fibreset=fibreset, obtitle=obtitle, obstemp=obstemp, progtemp=progtemp)
        obrealisation = OBRealisation(obid=obid, obstartmjd=obstart, obspec=obspec)
        exposure = Exposure(expmjd=expmjd, obrealisation=obrealisation)
        run = Run(runid=runid, armconfig=arm, exposure=exposure)
        return {'run': run, 'obrealisation': obrealisation, 'armconfig': arm}, {}

    def build_index(self) -> None:
        if self.index is None:
            self.index = pd.DataFrame({'cname': [i.cname for i in self.weavetargets]})
        super(HeaderFibinfoFile, self).build_index()

    def read_primary(self):
        return Header(fits.open(self.fname)[0].header, self.index)

    def read_fluxes(self):
        return Array(fits.open(self.fname)[1].data, self.index)

    def read_ivars(self):
        return Array(fits.open(self.fname)[2].data, self.index)

    def read_fluxes_noss(self):
        return Array(fits.open(self.fname)[3].data, self.index)

    def read_ivars_noss(self):
        return Array(fits.open(self.fname)[4].data, self.index)

    def read_sens_funcs(self):
        return Array(fits.open(self.fname)[5].data, self.index)

    def _read_fibtable(self):
        return AstropyTable(fits.open(self.fname)[self.fibinfo_i].data)

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
    parents = [OBRealisation, ArmConfig]
    constructed_from = [L1Single]
    indexer = 'armconfig'

    @classmethod
    def match(cls, directory):
        return directory.glob('stacked_*.fit')


class L1SuperStack(File):
    parents = [OBSpec, ArmConfig]
    indexer = 'armconfig'
    constructed_from = [L1Single]

    @classmethod
    def match(cls, directory):
        return directory.glob('superstacked_*.fit')


class L1SuperTarget(File):
    parents = [ArmConfig, WeaveTarget]
    factors = ['binning', 'mode']
    indexers = 'armconfig'
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
    parents = [Multiple(ArmConfig, 1, 3), FibreSet]
    factors = ['binning', 'mode']
    constructed_from = [Multiple(L1Stack, 0, 3), Multiple(L1SuperStack, 0, 3)]

    @classmethod
    def match(cls, directory):
        return directory.glob('(super)?stacked_*_aps.fit')


class L2SuperTarget(File):
    parents = [Multiple(ArmConfig, 1, 3), WeaveTarget]
    factors = ['mode', 'binning']
    constructed_from = [Multiple(L1SuperTarget, 2, 3)]

    @classmethod
    def match(cls, directory):
        return directory.glob('[Lm]?WVE_*_aps.fit')
