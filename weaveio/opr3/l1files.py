import logging
from pathlib import Path
from typing import Union, List, Tuple, Dict

from astropy.io import fits
from astropy.io.fits.hdu.base import _BaseHDU
from astropy.table import Table as AstropyTable
import numpy as np

from weaveio.config_tables import progtemp_config
from weaveio.file import File, PrimaryHDU, TableHDU, SpectralBlockHDU, SpectralRowableBlock, BinaryHDU
from weaveio.hierarchy import unwind, collect, Multiple, One2One, Hierarchy
from weaveio.opr3.hierarchy import Survey, SubProgramme, SurveyCatalogue, \
    WeaveTarget, SurveyTarget, Fibre, FibreTarget, ProgTemp, ArmConfig, ObsTemp, \
    OBSpec, OB, Exposure, Run, Observation, RawSpectrum, L1SingleSpectrum, L1StackSpectrum, L1SuperStackSpectrum, L1SuperTargetSpectrum, CASU, WavelengthHolder
from weaveio.writequery import groupby, CypherData


class HeaderFibinfoFile(File):
    is_template = True

    @classmethod
    def length(cls, path):
        return len(cls.read_fibinfo_dataframe(path))

    @classmethod
    def read_fibinfo_dataframe(cls, path, slc=None):
        hdus = fits.open(path)
        fibinfo_hdu = hdus['FIBTABLE']
        fibinfo = AstropyTable(fibinfo_hdu.data).to_pandas()
        fibinfo.columns = [i.lower() for i in fibinfo.columns]
        if 'nspec' in fibinfo.columns:
            fibinfo['spec_index'] = fibinfo['nspec'] - 1
        slc = slice(None) if slc is None else slc
        return fibinfo.iloc[slc]

    @classmethod
    def read_surveyinfo(cls, df_fibinfo):
        df_svryinfo = df_fibinfo[['targsrvy', 'targprog', 'targcat']].drop_duplicates()
        df_svryinfo['progid'] = df_svryinfo['targsrvy'] + df_svryinfo['targprog']
        df_svryinfo['catid'] = df_svryinfo['progid'] + df_svryinfo['targcat']
        return df_svryinfo

    @classmethod
    def read_fibretargets(cls, obspec, df_svryinfo, df_fibinfo):
        srvyinfo = CypherData(df_svryinfo)
        fibinfo = CypherData(df_fibinfo)
        with unwind(srvyinfo) as svryrow:
            with unwind(svryrow['targsrvy']) as surveyname:
                survey = Survey(name=surveyname)
            surveys = collect(survey)
            prog = SubProgramme(name=svryrow['targprog'], progid=svryrow['progid'], surveys=surveys)
            cat = SurveyCatalogue(name=svryrow['targcat'], catid=svryrow['catid'], subprogramme=prog)
        cat_collection = collect(cat)
        cats = groupby(cat_collection, 'name')
        with unwind(fibinfo) as fibrow:
            fibre = Fibre(fibreid=fibrow['fibreid'])  #  must be up here for some reason otherwise, there will be duplicates
            cat = cats[fibrow['targcat']]
            weavetarget = WeaveTarget(cname=fibrow['cname'])
            surveytarget = SurveyTarget(surveycatalogue=cat, weavetarget=weavetarget, tables=fibrow)
            fibtarget = FibreTarget(obspec=obspec, surveytarget=surveytarget, fibre=fibre, tables=fibrow)
        return collect(fibtarget, fibrow)

    @classmethod
    def read_distinct_survey_info(cls, df_svryinfo):
        rows = CypherData(df_svryinfo['targsrvy'].drop_duplicates().values, 'surveynames')
        with unwind(rows) as survey_name:
            survey = Survey(name=survey_name)
        survey_list = collect(survey)
        survey_dict = groupby(survey_list, 'name')

        # each row is (subprogramme, [survey_name, ...])
        s = df_svryinfo.groupby('progid')[['targsrvy', 'targprog']].aggregate(lambda x: x.values.tolist())
        rows = CypherData(s, 'targprog_rows')
        with unwind(rows) as row:
            with unwind(row['targsrvy']) as survey_name:
                survey = survey_dict[survey_name]
            surveys = collect(survey)
            subprogramme = SubProgramme(surveys=surveys, name=row['targprog'][0], progid=row['progid'])
        subprogramme_list = collect(subprogramme)
        subprogramme_dict = groupby(subprogramme_list, 'progid')

        # catalogue has 1 programme
        # programme can have many catalogues
        rows = CypherData(df_svryinfo[['targcat', 'progid', 'catid']].drop_duplicates(), 'targcat_rows')
        with unwind(rows) as row:
            catalogue = SurveyCatalogue(subprogramme=subprogramme_dict[row['progid']], name=row['targcat'], catid=row['catid'])
        catalogue_list = collect(catalogue)
        return survey_list, subprogramme_list, catalogue_list

    @classmethod
    def read_hierarchy(cls, header, df_svryinfo):
        surveys, subprogrammes, catalogues = cls.read_distinct_survey_info(df_svryinfo)
        runid = int(header['RUN'])
        camera = str(header['CAMERA'].lower()[len('WEAVE'):])
        expmjd = np.round(float(header['MJD-OBS']), 6)
        obstart = np.round(float(header['OBSTART']), 6)
        res = str(header['VPH']).rstrip('123').lower().replace('res', '')
        obtitle = str(header['OBTITLE'])
        xml = str(header['CAT-NAME']).replace('./', '') if 'xml' in header['CAT-NAME'] else str(header['INFILE'])  # stack files do this
        obid = float(header['OBID'])

        progtemp = ProgTemp.from_progtemp_code(header['PROGTEMP'])
        vph = int(progtemp_config[(progtemp_config['mode'] == progtemp.instrumentconfiguration.mode)
                                  & (progtemp_config['resolution'] == res.lower().replace('res', ''))][f'{camera}_vph'].iloc[0])
        arm = ArmConfig(vph=vph, resolution=res, camera=camera)  # must instantiate even if not used
        obstemp = ObsTemp.from_header(header)
        obspec = OBSpec(xml=xml, obtitle=obtitle, obstemp=obstemp, progtemp=progtemp,
                        surveycatalogues=catalogues, subprogrammes=subprogrammes, surveys=surveys)
        ob = OB(obid=obid, obstartmjd=obstart, obspec=obspec)
        exposure = Exposure(expmjd=expmjd, ob=ob)
        run = Run(runid=runid, armconfig=arm, exposure=exposure)
        observation = Observation.from_header(run, header)
        return {'run': run, 'ob': ob, 'obspec': obspec, 'armconfig': arm, 'observation': observation}

    @classmethod
    def read_schema(cls, path: Path, slc: slice = None):
        header = cls.read_header(path)
        fibinfo = cls.read_fibinfo_dataframe(path, slc)
        srvyinfo = cls.read_surveyinfo(fibinfo)
        hiers = cls.read_hierarchy(header, srvyinfo)
        fibretarget_collection, fibrows = cls.read_fibretargets(hiers['obspec'], srvyinfo, fibinfo)
        return hiers, header, fibinfo, fibretarget_collection, fibrows

    @classmethod
    def read_header(cls, path: Path, i=0):
        return fits.open(path)[i].header

    @classmethod
    def read_fibtable(cls, path: Path):
        return AstropyTable(fits.open(path)[cls.fibinfo_i].data)

    @classmethod
    def read(cls, directory: Path, fname: Path, slc: slice = None) -> 'File':
        raise NotImplementedError


class RawFile(HeaderFibinfoFile):
    match_pattern = 'r[0-9]+\.fit'
    hdus = {'primary': PrimaryHDU, 'counts1': SpectralBlockHDU, 'counts2': SpectralBlockHDU, 'fibtable': TableHDU, 'guidinfo': TableHDU, 'metinfo': TableHDU}
    parents = [CASU]
    produces = [One2One(Observation), One2One(RawSpectrum)]

    @classmethod
    def fname_from_runid(cls, runid):
        return f'r{runid:07.0f}.fit'

    @classmethod
    def read(cls, directory: Union[Path, str], fname: Union[Path, str], slc: slice = None):
        abspath = Path(directory) / Path(fname)
        relpath = str(abspath.relative_to(directory))
        hiers, header, fibinfo, fibretarget_collection, fibrow_collection = cls.read_schema(abspath, slc)
        observation = hiers['observation']
        hdus, file, _ = cls.read_hdus(directory, fname, casu=observation.casu)
        raw = RawSpectrum(sourcefile=relpath, casu=observation.casu, observation=observation, nrow=-1)
        raw.attach_products(file, **hdus)
        observation.attach_products(file, **hdus)
        return file


class L1File(HeaderFibinfoFile):
    is_template = True
    hdus = {'primary': PrimaryHDU, 'flux': SpectralRowableBlock, 'ivar': SpectralRowableBlock,
            'flux_noss': SpectralRowableBlock, 'ivar_noss': SpectralRowableBlock,
            'sensfunc': SpectralRowableBlock, 'fibtable': TableHDU}
    parents = [One2One(WavelengthHolder)]

    @classmethod
    def wavelengths(cls, rootdir: Path, fname: str):
        hdulist = fits.open(rootdir / fname)
        header = hdulist[1].header
        increment, zeropoint, size = header['cd1_1'], header['crval1'], header['naxis1']
        return WavelengthHolder(wvl=(np.arange(0, size) * increment) + zeropoint,
                                cd1_1=header['cd1_1'], crval1=header['crval1'], naxis1=header['naxis1'])

    @classmethod
    def read_hdus(cls, directory: Union[Path, str], fname: Union[Path, str],
                  **hierarchies: Union[Hierarchy, List[Hierarchy]]) -> Tuple[Dict[str, 'HDU'], 'File', List[_BaseHDU]]:
        return super().read_hdus(directory, fname, **hierarchies, wavelengthholder=cls.wavelengths(directory, fname))


class L1SingleFile(L1File):
    match_pattern = 'single_\d+\.fit'
    parents = L1File.parents + [One2One(RawFile), CASU]
    produces = [L1SingleSpectrum]
    version_on = ['rawfile']

    @classmethod
    def fname_from_runid(cls, runid):
        return f'single_{runid:07.0f}.fit'

    @classmethod
    def read(cls, directory: Union[Path, str], fname: Union[Path, str], slc: slice = None):
        fname = Path(fname)
        directory = Path(directory)
        path = directory / fname
        hiers, header, fibinfo, fibretarget_collection, fibrow_collection = cls.read_schema(path, slc)
        observation = hiers['observation']
        casu = observation.casu
        inferred_raw_path = fname.with_name(fname.name.replace('single_', 'r'))
        matched_files = list(directory.rglob(inferred_raw_path.name))
        if not matched_files:
            logging.warning(f"{fname} does not have a matching raw file. "
                            f"The database will create the link but the rawspectra will not be available until the missing file is.")
        elif len(matched_files) > 1:
            raise FileExistsError(f"Whilst searching for {inferred_raw_path.name}, we found more than one match:"
                                  f"\n{matched_files}")
        else:
            inferred_raw_path = matched_files[0]
        inferred_raw_path = inferred_raw_path.relative_to(directory)
        raw = RawSpectrum(sourcefile=str(inferred_raw_path), casu=casu, observation=observation, nrow=-1)
        rawfile = RawFile(fname=str(inferred_raw_path.name), path=inferred_raw_path, casu=casu)  # merge this one instead of finding, then we can start from single or raw files
        hdus, file, _ = cls.read_hdus(directory, fname, rawfile=rawfile, casu=casu)
        with unwind(fibretarget_collection, fibrow_collection) as (fibretarget, fibrow):
            single_spectrum = L1SingleSpectrum(sourcefile=str(inferred_raw_path.name),
                                               nrow=fibrow['nspec'],
                                               rawspectrum=raw, fibretarget=fibretarget,
                                               casu=casu, tables=fibrow)
            single_spectrum.attach_products(file, index=fibrow['spec_index'], **hdus)
        single_spectra = collect(single_spectrum)  # must collect at the end
        return file


class L1StackedBaseFile(L1File):
    is_template = True

    @classmethod
    def parent_runids(cls, path):
        header = cls.read_header(path, 0)
        runids = [int(v) for k, v in header.items() if k.startswith('RUNS0')]
        if len(runids) == 0:
            header = cls.read_header(path, 1)
            runids = [int(v.split('.')[0].split('_')[1]) for k, v in header.items() if k.startswith('PROV0') and 'formed' not in v]
        return runids

    @classmethod
    def get_provenance(cls, path):
        header = fits.open(path)[0].header
        vals = {k: header[k] for k in header.keys() if k.startswith('PROV')}
        return [v.split('[')[0] for k, v in vals.items() if int(k[len('PROV'):]) > 0 ]



class L1StackFile(L1StackedBaseFile):
    match_pattern = 'stack_[0-9]+\.fit'
    produces = [L1StackSpectrum]
    parents = L1StackedBaseFile.parents + [Multiple(L1SingleFile), OB, ArmConfig, CASU]

    @classmethod
    def fname_from_runid(cls, runid):
        return f'stack_{runid:07.0f}.fit'

    @classmethod
    def get_single_files(cls, absdirectory: Path, fname: Path):
        l1singlefiles = []
        paths = []
        single_fnames = cls.get_provenance(absdirectory / fname)
        for single_fname in single_fnames:
            file = L1SingleFile.find(fname=single_fname)
            l1singlefiles.append(file)
            paths.append(file.node.get('path'))
        return l1singlefiles, paths

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

        observation = hiers['observation']
        ob = hiers['ob']
        armconfig = hiers['armconfig']
        casu = observation.casu
        singlefiles, sourcefiles = cls.get_single_files(directory, fname)
        hdus, file, _ = cls.read_hdus(directory, fname, l1singlefiles=singlefiles, ob=ob,
                                   armconfig=armconfig, casu=casu)
        with unwind(fibretarget_collection, fibrow_collection) as (fibretarget, fibrow):
            single_spectra = []
            for singlefile, sourcefile in zip(singlefiles, sourcefiles):
                single_spectrum = L1SingleSpectrum.find(sourcefile=sourcefile, nrow=fibrow['nspec'])
                single_spectra.append(single_spectrum)
            stack_spectrum = L1StackSpectrum(sourcefile=str(path.relative_to(directory)), nrow=fibrow['nspec'],
                                             l1singlespectra=single_spectra, ob=ob,
                                             armconfig=armconfig, fibretarget=fibretarget,
                                             casu=casu, tables=fibrow)
            stack_spectrum.attach_products(file, index=fibrow['spec_index'], **hdus)
        stack_spectra = collect(stack_spectrum)  # must collect at the end
        return file


class L1SuperStackFile(L1StackedBaseFile):
    match_pattern = 'superstack_[0-9]+\.fit'
    produces = [L1SuperStackSpectrum]
    parents = L1StackedBaseFile.parents + [Multiple(L1SingleFile), OBSpec, ArmConfig, CASU]

    @classmethod
    def fname_from_runid(cls, runid):
        return f'superstack_{runid:07.0f}.fit'

    @classmethod
    def match(cls, directory):
        return directory.glob()

    @classmethod
    def read(cls, directory: Union[Path, str], fname: Union[Path, str], slc: slice = None):
        raise NotImplementedError


class L1SuperTargetFile(L1StackedBaseFile):
    match_pattern = 'WVE_.+\.fit'
    parents = L1StackedBaseFile.parents + [WeaveTarget, Multiple(L1SingleFile, 2), CASU]
    produces = [L1SuperTargetSpectrum]
    recommended_batchsize = None

    @classmethod
    def read(cls, directory: Union[Path, str], fname: Union[Path, str], slc: slice = None):
        raise NotImplementedError


