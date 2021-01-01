from pathlib import Path
from typing import Union

from astropy.io import fits
from astropy.table import Table as AstropyTable

from weaveio.config_tables import progtemp_config
from weaveio.file import File, PrimaryHDU, TableHDU, SpectralBlockHDU, SpectralRowableBlock, BinaryHDU
from weaveio.hierarchy import unwind, collect, Multiple
from weaveio.opr3.hierarchy import Survey, SubProgramme, SurveyCatalogue, \
    WeaveTarget, SurveyTarget, Fibre, FibreTarget, ProgTemp, ArmConfig, ObsTemp, \
    OBSpec, OB, Exposure, Run, Observation, RawSpectrum, L1SingleSpectrum, L1StackSpectrum, L1SuperStackSpectrum, L1SuperTargetSpectrum, CASU
from weaveio.writequery import groupby, CypherData


class HeaderFibinfoFile(File):
    is_template = True

    @classmethod
    def read_fibinfo_dataframe(cls, fname, slc=None):
        hdus = fits.open(fname)
        fibinfo_hdu = [i for i in hdus if i.name == 'FIBTABLE'][0]
        fibinfo = AstropyTable(fibinfo_hdu.data).to_pandas()
        fibinfo.columns = [i.lower() for i in fibinfo.columns]
        if 'nspec' in fibinfo.columns:
            fibinfo['spec_index'] = fibinfo['nspec'] - 1
        fibinfo = fibinfo[fibinfo.cname != '']  # exclude parked or disabled fibres
        slc = slice(None) if slc is None else slc
        return fibinfo.iloc[slc]

    @classmethod
    def read_surveyinfo(cls, df_fibinfo):
        return df_fibinfo[['targsrvy', 'targprog', 'targcat']].drop_duplicates()

    @classmethod
    def read_fibretargets(cls, obspec, df_svryinfo, df_fibinfo):
        srvyinfo = CypherData(df_svryinfo)
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
            fibre = Fibre(fibreid=fibrow['fibreid'])  #  must be up here for some reason otherwise, there will be duplicates
            cat = cats[fibrow['targcat']]
            weavetarget = WeaveTarget(cname=fibrow['cname'])
            surveytarget = SurveyTarget(surveycatalogue=cat, weavetarget=weavetarget, tables=fibrow)
            fibtarget = FibreTarget(obspec=obspec, surveytarget=surveytarget, fibre=fibre, tables=fibrow)
        return collect(fibtarget, fibrow)

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
    def read_schema(cls, path: Path, slc: slice = None):
        header = cls.read_header(path)
        fibinfo = cls.read_fibinfo_dataframe(path, slc)
        hiers = cls.read_hierarchy(header)
        srvyinfo = cls.read_surveyinfo(fibinfo)
        fibretarget_collection, fibrows = cls.read_fibretargets(hiers['obspec'], srvyinfo, fibinfo)
        return hiers, header, fibinfo, fibretarget_collection, fibrows

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
    match_pattern = 'r*.fit'
    hdus = {'primary': PrimaryHDU, 'counts1': SpectralBlockHDU, 'counts2': SpectralBlockHDU, 'fibtable': TableHDU, 'guidinfo': TableHDU, 'metinfo': TableHDU}
    produces = [RawSpectrum, Observation]

    @classmethod
    def hash_spectrum(cls, path: Path, fname: Path):
        return str(fname)

    @classmethod
    def fname_from_runid(cls, runid):
        return f'r{runid:07.0f}.fit'

    @classmethod
    def read(cls, directory: Union[Path, str], fname: Union[Path, str], slc: slice = None):
        path = Path(directory) / Path(fname)
        hiers, header, fibinfo, fibretarget_collection, fibrow_collection = cls.read_schema(path)
        observation = hiers['observation']
        hdus, file = cls.read_hdus(directory, fname)
        hashid = cls.hash_spectrum(path, fname)
        raw = RawSpectrum(hashid=hashid, casu=observation.casu, observation=observation)
        raw.attach_products(file, **hdus)
        observation.attach_products(file, **hdus)
        return file


class L1File(HeaderFibinfoFile):
    is_template = True
    hdus = {'primary': PrimaryHDU, 'flux': SpectralRowableBlock, 'ivar': SpectralRowableBlock,
            'flux_noss': SpectralRowableBlock, 'ivar_noss': SpectralRowableBlock,
            'sensfunc': BinaryHDU, 'fibtable': TableHDU}

    @classmethod
    def hash_spectra(cls, path, fname, slc: slice = None):
        fibinfo_df = cls.read_fibinfo_dataframe(path, slc)
        return [f'{fname}[{row.nspec}]' for _, row in fibinfo_df.iterrows()]


class L1SingleFile(L1File):
    match_pattern = 'single_*.fit'
    parents = [RawFile]
    produces = [L1SingleSpectrum]

    @classmethod
    def fname_from_runid(cls, runid):
        return f'single_{runid:07.0f}.fit'

    @classmethod
    def read(cls, directory: Union[Path, str], fname: Union[Path, str], slc: slice = None):
        fname = Path(fname)
        directory = Path(directory)
        path = directory / fname
        hiers, header, fibinfo, fibretarget_collection, fibrow_collection = cls.read_schema(path)
        observation = hiers['observation']
        casu = observation.casu
        inferred_raw_fname = fname.with_name(fname.name.replace('single_', 'r'))
        raw = RawSpectrum(hashid=RawFile.hash_spectrum(path, inferred_raw_fname),
                          casu=casu, observation=observation)
        rawfile = RawFile(inferred_raw_fname)
        hdus, file = cls.read_hdus(directory, fname, rawfile=rawfile)
        hashes = CypherData(cls.hash_spectra(path, fname), 'hashes')
        with unwind(fibretarget_collection, fibrow_collection, hashes) as (fibretarget, fibrow, hsh):
            single_spectrum = L1SingleSpectrum(hashid=hsh, rawspectrum=raw, fibretarget=fibretarget,
                                               casu=casu, tables=fibrow)
            single_spectrum.attach_products(file, index=fibrow['spec_index'], **hdus)
        single_spectra = collect(single_spectrum)  # must collect at the end
        return file


class L1StackedBaseFile(L1File):
    is_template = True


class L1StackFile(L1StackedBaseFile):
    match_pattern = 'stacked_*.fit'
    produces = [L1StackSpectrum]
    parents = [Multiple(L1SingleFile), OB, ArmConfig, CASU]

    @classmethod
    def parent_runids(cls, path):
        header = cls.read_header(path)
        return [int(v) for k, v in header.items() if k.startswith('RUNS0')]

    @classmethod
    def get_single_files(cls, path):
        l1singlefiles = []
        runids = cls.parent_runids(path)
        for runid in runids:
            raw_fname = RawFile.fname_from_runid(runid)
            single_fname = L1SingleFile.fname_from_runid(runid)
            raw = RawFile(fname=raw_fname)
            l1singlefiles.append(L1SingleFile(single_fname, rawfile=raw))
        return l1singlefiles

    @classmethod
    def read(cls, directory: Union[Path, str], fname: Union[Path, str], slc: slice = None):
        """
        L1Stack inherits everything from the lowest numbered single/raw files so we are missing data,
        therefore we require that all the referenced Runs are present before loading in
        """
        fname = Path(fname)
        directory = Path(directory)
        path = directory / fname
        hiers, header, fibinfo, fibretarget_collection, fibrow_collection = cls.read_schema(path, slc)
        hashes = cls.hash_spectra(path, fname, slc)

        observation = hiers['observation']
        ob = hiers['ob']
        armconfig = hiers['armconfig']
        casu = observation.casu
        hashes = CypherData(hashes, 'hashes')
        singlefiles = cls.get_single_files(path)
        hdus, file = cls.read_hdus(directory, fname, l1singlefiles=singlefiles, ob=ob,
                                   armconfig=armconfig, casu=casu)
        with unwind(fibretarget_collection, fibrow_collection, hashes) as (fibretarget, fibrow, hsh):
            single_spectra = []
            for singlefile in singlefiles:
                single_spectrum = L1SingleSpectrum.find(anonymous_parents=[fibretarget, singlefile, casu])
                single_spectra.append(single_spectrum)
            stack_spectrum = L1StackSpectrum(hashid=hsh, l1singlespectra=single_spectra, ob=ob,
                                             armconfig=armconfig, fibretarget=fibretarget,
                                             casu=casu, tables=fibrow)
            stack_spectrum.attach_products(file, index=fibrow['spec_index'], **hdus)
        stack_spectra = collect(stack_spectrum)  # must collect at the end
        return file


class L1SuperStackFile(L1StackedBaseFile):
    match_pattern = 'superstacked_*.fit'
    produces = [L1SuperStackSpectrum]
    parents = [Multiple(L1SingleFile), OBSpec, ArmConfig, CASU]

    @classmethod
    def match(cls, directory):
        return directory.glob()

    @classmethod
    def read(cls, directory: Union[Path, str], fname: Union[Path, str], slc: slice = None):
        raise NotImplementedError


class L1SuperTargetFile(L1StackedBaseFile):
    match_pattern = 'WVE_*.fit'
    produces = [L1SuperTargetSpectrum]

    @classmethod
    def read(cls, directory: Union[Path, str], fname: Union[Path, str], slc: slice = None):
        raise NotImplementedError


