from collections import defaultdict
from pathlib import Path
from typing import Tuple

from astropy.io import fits
from astropy.table import Table as AstropyTable

from weaveio.config_tables import progtemp_config
from weaveio.file import File
from weaveio.graph import Graph
from weaveio.hierarchy import Multiple, unwind, collect
from weaveio.oldproduct import Header, Array, Table
from weaveio.opr3.hierarchy import Survey, SubProgramme, SurveyCatalogue, \
    WeaveTarget, SurveyTarget, Fibre, FibreTarget, ProgTemp, ArmConfig, ObsTemp,\
    OBSpec, OB, Exposure, Run, L1SingleSpectrum, RawSpectrum, L1StackSpectrum, \
    L1SuperStackSpectrum, L1SuperTargetSpectrum
from weaveio.writequery import groupby, CypherData


class HeaderFibinfoFile(File):
    is_template = True
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

    @classmethod
    def make_fibretargets(cls, df_svryinfo, df_fibinfo):
        srvyinfo = CypherData(df_svryinfo)  # surveys split inline
        fibinfo = CypherData(df_fibinfo)
        with unwind(srvyinfo) as svryrow:
            with unwind(svryrow['targsrvy']) as surveyname:
                survey = Survey(surveyname=surveyname)
            surveys = collect(survey)
            prog = SubProgramme(targprog=svryrow['targprog'], surveys=surveys)
            cat = SurveyCatalogue(targcat=svryrow['targcat'], subprogramme=prog)
        cat_collection = collect(cat)
        cats = groupby(cat_collection, 'targcat')
        with unwind(fibinfo) as fibrow:
            cat = cats[fibrow['targcat']]
            weavetarget = WeaveTarget(cname=fibrow['cname'])
            surveytarget = SurveyTarget(surveycatalogue=cat, weavetarget=weavetarget, tables=fibrow)
            fibre = Fibre(fibreid=fibrow['fibreid'])
            fibtarget = FibreTarget(surveytarget=surveytarget, fibre=fibre, tables=fibrow)
        return collect(fibtarget)

    @classmethod
    def read_fibinfo_dataframe(cls, fname):
        fibinfo = AstropyTable(fits.open(fname)[cls.fibinfo_i].data).to_pandas().sort_values('FIBREID')
        fibinfo.columns = [i.lower() for i in fibinfo.columns]
        return fibinfo

    @classmethod
    def read_surveyinfo(cls, df_fibinfo):
        return df_fibinfo[['targsrvy', 'targprog', 'targcat']].drop_duplicates()

    @classmethod
    def read_hierarchy(cls, directory: Path, fname: Path):
        fullpath = directory / fname
        header = fits.open(fullpath)[0].header
        runid = int(header['RUN'])
        camera = str(header['CAMERA'].lower()[len('WEAVE'):])
        expmjd = str(header['MJD-OBS'])
        res = str(header['VPH']).rstrip('123')
        obstart = str(header['OBSTART'])
        obtitle = str(header['OBTITLE'])
        xml = str(header['CAT-NAME'])
        obid = str(header['OBID'])

        progtemp = ProgTemp.from_progtemp_code(header['PROGTEMP'])
        vph = int(progtemp_config[(progtemp_config['mode'] == progtemp.instrumentconfiguration.mode)
                                  & (progtemp_config['resolution'] == res)][f'{camera}_vph'].iloc[0])
        arm = ArmConfig(vph=vph, resolution=res, camera=camera)  # must instantiate even if not used
        obstemp = ObsTemp.from_header(header)

        cls.make_surveys(fullpath)
        with cls.each_fibinfo_row(fullpath) as row:
            fibretarget = cls.each_fibretarget(row)
            fibreset = FibreSet(fibretargets=fibretarget)
        obspec = OBSpec(xml=xml, fibreset=fibreset, obtitle=obtitle, obstemp=obstemp, progtemp=progtemp)
        obrealisation = OB(obid=obid, obstartmjd=obstart, obspec=obspec)
        exposure = Exposure(expmjd=expmjd, obrealisation=obrealisation)
        run = Run(runid=runid, armconfig=arm, exposure=exposure)
        return {'run': run, 'obrealisation': obrealisation, 'obspec': obspec, 'armconfig': arm}


    @classmethod
    def read_l1single(cls, raw, fibrow):
        fibretarget = cls.each_fibretarget(fibrow)
        return L1SingleSpectrum(fibretarget=fibretarget, raw=raw, findex=fibrow['_input_index'])

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


class RawFile(HeaderFibinfoFile):
    parents = [RawSpectrum]
    fibinfo_i = 3
    match_pattern = 'r*.fit'

    @classmethod
    def read(cls, directory: Path, fname: Path):
        hiers = cls.read_hierarchy(directory, fname)
        raw = RawSpectrum(run=hiers['run'])
        return  cls(fname=fname, raw=raw)


class L1SingleFile(HeaderFibinfoFile):
    parents = [Multiple(L1SingleSpectrum, constrain=[RawSpectrum])]
    match_pattern = 'single_*.fit'

    @classmethod
    def read(cls, directory: Path, fname: Path):
        """L1Single inherits all header data from raw so we can construct both when needed"""
        hiers = cls.read_hierarchy(directory, fname)
        inferred_raw_fname = fname.with_name(fname.name.replace('single', 'r'))
        hiers['raw'] = raw = RawSpectrum(run=hiers['run'])
        inferred_rawfile = RawFile(inferred_raw_fname, **hiers)
        hiers['rawfile'] = inferred_rawfile
        with cls.each_fibinfo_row(fname) as row:
            single_spectra = cls.read_l1single(raw, row)
        return cls(fname=fname, l1singlespectra=single_spectra)


class L1StackFile(HeaderFibinfoFile):
    parents = [Multiple(L1StackSpectrum, constrain=[Exposure, ArmConfig])]
    dependencies = [Run]
    match_pattern = 'stacked_*.fit'

    @classmethod
    def make_spectra_from_single(cls, singles):
        try:
            spec_type = cls.parents[0].node
        except AttributeError:
            spec_type = cls.parents[0]
        return {spec_type.singular_name: spec_type(l1singlespectra=singles)}

    @classmethod
    def read(cls, directory: Path, fname: Path):
        """
        L1Stack inherits everything from the lowest numbered single/raw files so we are missing data,
        therefore we require that all the referenced Runs are present before loading in
        """
        fullpath = directory / fname
        hiers = cls.read_hierarchy(directory, fname)
        armconfig = hiers['armconfig']
        header = fits.open(fullpath)[0].header
        runids = sorted([(int(k[len('RUNS'):]), v) for k, v in header.items() if k.startswith('RUNS')], key=lambda x: x[1])
        # start at Run, assuming it has already been created
        runs = [Run(runid=rid, armconfig=armconfig, exposure=None) for rid in runids]
        raws = [RawSpectrum(run=run) for run in runs]
        with cls.each_fibinfo_row(fname) as row:
            # this list comp is not a loop in cypher but simply a sequence of statements one after the other
            singlespectra = [cls.read_l1single(raw, row) for raw in raws]
            data = cls.make_spectra_from_single(singlespectra)  # and this is per fibretarget
        return cls(fname=fname, **data)


class L1SuperStackFile(L1StackFile):
    parents = [Multiple(L1SuperStackSpectrum, constrain=[OBSpec, ArmConfig])]
    match_pattern = 'superstacked_*.fit'

    @classmethod
    def match(cls, directory):
        return directory.glob()


class L1SuperTargetFile(L1StackFile):
    parents = [L1SuperTargetSpectrum]
    match_pattern = 'WVE_*.fit'


class L2HeaderFibinfoFile(HeaderFibinfoFile):
    def read(self):
        """
        L2Singles inherit from L1Single and so if the other is missing we can fill most of it in
        """
        hiers, factors = super().read()
        header = fits.open(self.fname)[0].header
        runids = sorted([(int(k[len('RUNS'):]), v) for k, v in header.items() if k.startswith('RUNS')], key=lambda x: x[1])
        raw_fname = self.fname.with_name(f'r{runids[0]}.fit')
        single_fname = self.fname.with_name(f'single_{runids[0]}.fit')
        raw = RawFile(raw_fname, **hiers, **factors)
        single = L1SingleFile(single_fname, run=hiers['run'], raw=raw)
        armconfig = single.run.armconfig
        if armconfig.camera == 'red':
            camera = 'blue'
        else:
            camera = 'red'
        other_armconfig = ArmConfig(armcode=None, resolution=None, vph=None, camera=camera, colour=None)
        other_raw_fname = self.fname.with_name(f'r{runids[1]}.fit')
        other_single_fname = self.fname.with_name(f'single_{runids[1]}.fit')
        other_run = Run(runid=runids[1], run=single.run, armconfig=other_armconfig, exposure=single.run.exposure)
        other_raw = RawFile(other_raw_fname, run=other_run)
        other_single = L1SingleFile(other_single_fname, run=other_run, raw=other_raw)
        return {'singles': [single, other_single], 'exposure': single.run.exposure}, {}


class L2File(HeaderFibinfoFile):
    match_pattern = '*aps.fit'
