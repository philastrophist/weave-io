from pathlib import Path
from typing import Union

from astropy.io import fits
from astropy.table import Table as AstropyTable

from weaveio.config_tables import progtemp_config
from weaveio.file import File, PrimaryHDU, TableHDU, SpectralBlockHDU, SpectralRowableBlock, DataHDU
from weaveio.hierarchy import unwind, collect, merge_relationship
from weaveio.opr3.hierarchy import Survey, SubProgramme, SurveyCatalogue, \
    WeaveTarget, SurveyTarget, Fibre, FibreTarget, ProgTemp, ArmConfig, ObsTemp, \
    OBSpec, OB, Exposure, Run, Observation, RawSpectrum, L1SingleSpectrum, L1StackSpectrum, L1SuperStackSpectrum, L1SuperTargetSpectrum
from weaveio.writequery import groupby, CypherData


class HeaderFibinfoFile(File):
    is_template = True
    fibinfo_i = -1

    @classmethod
    def read_fibinfo_dataframe(cls, fname):
        fibinfo = AstropyTable(fits.open(fname)[cls.fibinfo_i].data).to_pandas().sort_values('FIBREID')
        fibinfo.columns = [i.lower() for i in fibinfo.columns]
        return fibinfo

    @classmethod
    def read_surveyinfo(cls, df_fibinfo):
        return df_fibinfo[['targsrvy', 'targprog', 'targcat']].drop_duplicates()

    @classmethod
    def read_fibretargets(cls, obspec, df_svryinfo, df_fibinfo):
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
            fibtarget = FibreTarget(obspec=obspec, surveytarget=surveytarget, fibre=fibre, tables=fibrow)
        return collect(fibtarget)

    @classmethod
    def read_hierarchy(cls, header):
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
        obspec = OBSpec(xml=xml, obtitle=obtitle, obstemp=obstemp, progtemp=progtemp)
        ob = OB(obid=obid, obstartmjd=obstart, obspec=obspec)
        exposure = Exposure(expmjd=expmjd, ob=ob)
        run = Run(runid=runid, armconfig=arm, exposure=exposure)
        observation = Observation.from_header(run, header)
        return {'run': run, 'ob': ob, 'obspec': obspec, 'armconfig': arm, 'observation': observation}

    @classmethod
    def read_schema(cls, path: Path):
        header = cls.read_header(path)
        fibinfo = cls.read_fibinfo_dataframe(path)
        hiers = cls.read_hierarchy(header)
        srvyinfo = cls.read_surveyinfo(fibinfo)
        fibretarget_collection = cls.read_fibretargets(hiers['obspec'], srvyinfo, fibinfo)
        return hiers, header, fibinfo, fibretarget_collection

    @classmethod
    def read_header(cls, path: Path):
        return fits.open(path)[0].header

    @classmethod
    def read_fibtable(cls, path: Path):
        return AstropyTable(fits.open(path)[cls.fibinfo_i].data)

    @classmethod
    def read(cls, directory: Path, fname: Path) -> 'File':
        raise NotImplementedError



class RawFile(HeaderFibinfoFile):
    fibinfo_i = 3
    match_pattern = 'r*.fit'
    hdus = {'primary': PrimaryHDU, 'counts1': SpectralBlockHDU, 'counts2': SpectralBlockHDU, 'fibtable': TableHDU, 'guidinfo': TableHDU, 'metinfo': TableHDU}
    produces = [RawSpectrum]

    @classmethod
    def read(cls, directory: Union[Path, str], fname: Union[Path, str]):
        path = Path(directory) / Path(fname)
        hiers, header, fibinfo, fibretarget_collection = cls.read_schema(path)
        observation = hiers['observation']
        hdus, file = cls.read_hdus(directory, fname)
        hashid = header['checksum']
        raw = RawSpectrum(hashid=hashid, casu=observation.casu, observation=observation)
        raw.attach_products(file, **hdus)
        return file


class L1File(HeaderFibinfoFile):
    is_template = True
    hdus = {'primary': PrimaryHDU, 'flux': SpectralRowableBlock, 'ivar': SpectralRowableBlock,
            'flux_noss': SpectralRowableBlock, 'ivar_noss': SpectralRowableBlock, 'sensfunc': DataHDU, 'fibtable': TableHDU}


class L1SingleFile(L1File):
    match_pattern = 'single_*.fit'
    produces = [L1SingleSpectrum]

    @classmethod
    def read(cls, directory: Union[Path, str], fname: Union[Path, str]):
        path = Path(directory) / Path(fname)
        hiers, header, fibinfo, fibretarget_collection = cls.read_schema(path)



        # hiers = cls.read_hierarchy(directory, fname)
        # inferred_raw_fname = fname.with_name(fname.name.replace('single', 'r'))
        # hiers['raw'] = raw = RawSpectrum(run=hiers['run'])
        # inferred_rawfile = RawFile(inferred_raw_fname, **hiers)
        # hiers['rawfile'] = inferred_rawfile
        # with cls.each_fibinfo_row(fname) as row:
        #     single_spectra = cls.read_l1single(raw, row)
        # return cls(fname=fname, l1singlespectra=single_spectra)


class L1StackedFile(L1File):
    pass


class L1StackFile(L1StackedFile):
    match_pattern = 'stacked_*.fit'
    produces = [L1StackSpectrum]

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


class L1SuperStackFile(L1StackedFile):
    match_pattern = 'superstacked_*.fit'
    produces = [L1SuperStackSpectrum]

    @classmethod
    def match(cls, directory):
        return directory.glob()


class L1SuperTargetFile(L1StackedFile):
    match_pattern = 'WVE_*.fit'
    produces = [L1SuperTargetSpectrum]


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


