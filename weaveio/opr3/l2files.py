import sys
from collections import namedtuple, defaultdict
from pathlib import Path
from typing import Union, List, Dict, Type, Set, Tuple, Optional

import inspect

import numpy as np
from astropy.io import fits
from astropy.io.fits.hdu.base import _BaseHDU
from astropy.table import Table, Column

from weaveio.file import File, PrimaryHDU, TableHDU
from weaveio.graph import Graph
from weaveio.hierarchy import Multiple, unwind, collect, Hierarchy, find_branch
from weaveio.opr3.hierarchy import APS, OB, OBSpec, Exposure, WeaveTarget, _predicate, Run, ArmConfig, FibreTarget, Fibre
from weaveio.opr3.l1 import L1Spectrum, L1SingleSpectrum, L1StackSpectrum, L1SupertargetSpectrum
from weaveio.opr3.l2 import L2, L2Single, L2Stack, L2Superstack, L2Supertarget, IngestedSpectrum, Fit, ModelSpectrum, Redrock, \
    RVSpecfit, Ferre, PPXF, Gandalf, GandalfModelSpectrum, CombinedIngestedSpectrum, CombinedModelSpectrum, Template, RedshiftArray, IvarIngestedSpectrum, IvarCombinedIngestedSpectrum, MaskedCombinedIngestedSpectrum, GandalfEmissionModelSpectrum, GandalfCleanModelSpectrum, \
    GandalfCleanIngestedSpectrum, L2Product
from weaveio.opr3.l1files import L1File, L1SuperstackFile, L1StackFile, L1SingleFile, L1SupertargetFile
from weaveio.writequery import CypherData, CypherVariable
from weaveio.writequery.actions import string_append
from weaveio.writequery.base import CypherAppendStr


MAX_REDSHIFT_GRID_LENGTH = 5000

class MissingDataError(Exception):
    pass


def column_name_acronym_replacements(columns:List[str], *acronyms: str) -> Dict[str, str]:
    replacements = {}
    for column in columns:
        for a in acronyms:
            for aa in [f'_{a}', f'_{a}_', f'{a}_']:
                if aa in column:
                    replacements[column.replace(aa.lower(), '')] = column
    return replacements


def filter_products_from_table(table: Table, maxlength: int) -> Table:
    columns = []
    for i in table.colnames:
        value = table[i]
        if len(value.shape) == 2:
            if value.shape[1] > maxlength:
                continue
        columns.append(i)
    t = table[columns]
    for col in table.colnames:
        t.rename_column(col, col.lower())
    return t


FitSpecs = namedtuple('FitSpecs', ['individuals', 'individual_models', 'combined', 'combined_model', 'arm_codes', 'nrow'])
GandalfSpecs = namedtuple('GandalfSpecs', ['model', 'ingested', 'emission', 'clean_model', 'clean_ingested', 'nrow'])


class L2File(File):
    singular_name = 'l2file'
    is_template = True
    match_pattern = '.*APS.fits'
    antimatch_pattern = '.*cube.*'
    L2 = L2Product
    L1 = L1Spectrum
    parents = [Multiple(L1File, 2, 3), APS, Multiple(L2)]
    children = []
    parts = ['RR', 'RVS', 'FR', 'GAND', 'PPXF']
    hdus = {'primary': PrimaryHDU,
            'galaxy_spectra': TableHDU,
            'class_spectra': TableHDU,
            'stellar_spectra': TableHDU,
            'class_table': TableHDU,
            'stellar_table': TableHDU,
            'galaxy_table': TableHDU}

    @classmethod
    def length(cls, path):
        hdus = fits.open(path)
        return len(hdus[1].data)

    @classmethod
    def make_l2(cls, spectra, aps, nspec, **hiers):
        return cls.L2(**{cls.L1.plural_name: [spectra[i] for i in range(nspec)], 'aps': aps}, **hiers)

    @classmethod
    def decide_filetype(cls, l1filetypes: List[Type[File]]) -> Type[File]:
        l1precedence = [L1SingleFile, L1StackFile, L1SuperstackFile, L1SupertargetFile]
        l2precedence = [L2SingleFile, L2StackFile, L2SuperstackFile, L2SupertargetFile]
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
        header = cls.read_header_and_aps(path)[0]
        ftypes, _ = zip(*cls.parse_fname(header, fname, instantiate=False))
        return cls.decide_filetype(ftypes) is cls

    @classmethod
    def parser_ftypes_runids(cls, header):
        header_info = [header.get(f'L1_REF_{i}', '.').split('.')[0].split('_') for i in range(4)]
        ftypes_header, runids_header = zip(*[i for i in header_info if len(i) > 1])
        return ftypes_header, list(map(int, runids_header))

    @classmethod
    def parse_fname(cls, header, fname, instantiate=True) -> List[L1File]:
        """
        Return the L1File type and the expected filename that formed this l2 file
        """
        ftype_dict = {
            'single': L1SingleFile,
            'stacked': L1StackFile, 'stack': L1StackFile,
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
        ftypes_header, runids_header = cls.parser_ftypes_runids(header)
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
    def read_header_and_aps(cls, path):
        return fits.open(path)[0].header, fits.open(path)[1].header['APS_V']

    @classmethod
    def read_hdus(cls, directory: Union[Path, str], fname: Union[Path, str], l1files: List[L1File],
                  **hierarchies: Union[Hierarchy, List[Hierarchy]]) -> Tuple[Dict[int, 'HDU'], 'File', List[_BaseHDU]]:
        fdict = {p.plural_name: [] for p in cls.parents if isinstance(p, Multiple) and issubclass(p.node, L1File)} # parse the 1lfile types separately
        for f in l1files:
            fdict[f.plural_name].append(f)
        hierarchies.update(fdict)
        hierarchies[cls.L2.plural_name] = hierarchies['l2']
        del hierarchies['l2']
        return super().read_hdus(directory, fname, **hierarchies)

    @classmethod
    def get_l1_filenames(cls, header):
        return [v for k, v in header.items() if 'APS_REF' in k]

    @classmethod
    def get_all_fibreids(cls, path):
        aps_ids = set()
        for hdu in fits.open(path)[1:]:
            try:
                aps_ids |= set(hdu.data['APS_ID'].tolist())
            except KeyError:
                pass
        return sorted(aps_ids)

    @classmethod
    def attach_products_to_spectra(cls, specs: FitSpecs, formatter, hdu, names: Dict[str, str]):
        if specs.individual_models is not None:
            with unwind(specs.individual_models, specs.arm_codes, specs.nrow) as (spectra, arm_code, nrow):
                column_name = string_append(f'model_{formatter}_', arm_code)
                with unwind(spectra) as spectrum:
                    spectrum = ModelSpectrum.from_cypher_variable(spectrum)
                    spectrum.attach_product('flux', hdu, column_name=column_name, index=nrow)
        if specs.individuals is not None:
            with unwind(specs.individuals, specs.arm_codes, specs.nrow) as (spectra, arm_code, nrow):
                for product in spectrum.products:
                    column_name = string_append(f'{names.get(product, product)}_{formatter}_', arm_code)
                    with unwind(spectra) as spectrum:
                        spectrum = IngestedSpectrum.from_cypher_variable(spectrum)
                        spectrum.attach_product(product, hdu, column_name=column_name, index=nrow)
        if specs.combined_model is not None:
            with unwind(specs.combined_model) as combined_model:
                spectrum = CombinedModelSpectrum.from_cypher_variable(combined_model)
                column_name = f'model_{formatter}_C'
                spectrum.attach_product('flux', hdu, column_name=column_name, index=specs.nrow)
        if specs.combined is not None:
            with unwind(specs.combined) as combined:
                spectrum = CombinedIngestedSpectrum.from_cypher_variable(combined)
                for product in spectrum.products:
                    column_name = f'{names.get(product, product)}_{formatter}_C'
                    spectrum.attach_product(product, hdu, column_name=column_name, index=specs.nrow)

    @classmethod
    def attach_products_to_gandalf_extra_spectra(cls, specs: GandalfSpecs, hdu):
        specs.emission.attach_product('flux', hdu, column_name=f'emission_GAND', index=specs.nrow, ignore_missing=True)
        specs.clean_model.emission.attach_product('flux', hdu, column_name=f'model_GAND', index=specs.nrow, ignore_missing=True)
        specs.clean_ingested.emission.attach_product('flux', hdu, column_name=f'model_clean_GAND', index=specs.nrow, ignore_missing=True)

    @classmethod
    def make_redrock_fit(cls, specs, row, nl1specs, replacements):
        templates = {}
        for template_name in Redrock.template_names:
            zs = row[f'CZZ_{template_name.upper()}']
            chi2s = row[f'CZZ_CHI2_{template_name.upper()}']
            start, end, step = [row[f'CZZ_{template_name.upper()}_{i}'] for i in ['START', 'END', 'STEP']]
            redshift_array = RedshiftArray(start=start, end=end, step=step, value=zs)
            template = Template(model_spectra=specs.individual_models, combined_model_spectrum=specs.combined_model,
                                redshift_array=redshift_array, name=template_name, chi2_array=chi2s)
            templates[template_name] = template
        return Redrock(model_spectra=[specs.individual_models[i] for i in range(nl1specs)],
                       combined_model_spectrum=specs.combined_model, tables=row, tables_replace=replacements, **templates)

    @classmethod
    def make_gandalf_structure(cls, specs, row, this_fname, nrow, replacements):
        model, ingested = specs.combined_model, specs.combined
        emission = GandalfEmissionModelSpectrum(sourcefile=this_fname, nrow=nrow, name='emission', arm_code='C',
                                                gandalf_model_spectrum=model)
        clean_model = GandalfCleanModelSpectrum(sourcefile=this_fname, nrow=nrow, name='clean_model', arm_code='C',
                                                gandalf_model_spectrum=model)
        clean_ingested = GandalfCleanIngestedSpectrum(sourcefile=this_fname, nrow=nrow, name='clean_ingested', arm_code='C',
                                                      combined_ingested_spectrum=ingested)
        gandalf = Gandalf(gandalf_model_spectrum=specs.combined_model, tables=row, tables_replace=replacements)
        return gandalf, GandalfSpecs(model=model, ingested=ingested, emission=emission, clean_model=clean_model,
                                     clean_ingested=clean_ingested, nrow=specs.nrow)

    @classmethod
    def read_l2product_table(cls, this_fname, spectrum_hdu, row: CypherVariable, nrow,
                             parent_l1filenames,
                             IngestedSpectrumClass: Optional[Type[IngestedSpectrum]],
                             CombinedIngestedSpectrumClass: Optional[Type[CombinedIngestedSpectrum]],
                             ModelSpectrumClass: Optional[Type[ModelSpectrum]],
                             CombinedModelSpectrumClass: Optional[Type[CombinedModelSpectrum]],
                             uses_disjoint_spectra: bool,
                             uses_combined_spectrum: Optional[bool],
                             formatter: str,
                             aps):
        # if the joint spectrum is not available, we dont read it, obvs
        if uses_combined_spectrum is None:
            uses_combined_spectrum = any('_C' in i for i in spectrum_hdu.data.names)
        fibre = Fibre.find(id=row['APS_ID'])
        with unwind(CypherData(parent_l1filenames)) as l1_fname:  # for each parent l1 file
            l1file = L1File.find(fname=l1_fname)
            _, fibretarget, l1spectrum, _ = find_branch(fibre, FibreTarget, L1Spectrum, l1file)
            arm_code = ArmConfig.find(anonymous_children=[l1spectrum])['arm_code']
            if uses_disjoint_spectra:
                individual = IngestedSpectrumClass(sourcefile=this_fname, nrow=nrow, name=formatter,
                                                   l1_spectrum=l1spectrum, aps=aps, arm_code=arm_code)
                individual_model = ModelSpectrumClass(sourcefile=this_fname, nrow=nrow, name=formatter,
                                                      arm_code=arm_code, ingested_spectrum=individual)
        # now collect spec and models relative to the fibretarget
        if uses_disjoint_spectra:
            l1files, l1spectra, fibretargets, arm_codes, individuals, individual_models = collect(l1file, l1spectrum, fibretarget,
                                                                                    arm_code, individual, individual_model)
        else:
            l1files, l1spectra, fibretargets, arm_codes = collect(l1file, l1spectrum, fibretarget,  arm_code)
            individuals, individual_models = None, None
        if uses_combined_spectrum:
            combined = CombinedIngestedSpectrumClass(sourcefile=this_fname, nrow=nrow,
                                                     l1_spectra=[l1spectra[i] for i in range(len(parent_l1filenames))],
                                                     arm_code='C', name=formatter, aps=aps)
            combined_model = CombinedModelSpectrumClass(sourcefile=this_fname, nrow=nrow,
                                                        combined_ingested_spectrum=combined,
                                                        arm_code='C', name=formatter)
        else:
            combined, combined_model = None, None
        return FitSpecs(individuals, individual_models, combined, combined_model, arm_codes, nrow), l1spectra, fibretargets

    @classmethod
    def read_redrock(cls, this_fname, spectrum_hdu, colnames, safe_table: CypherVariable, parent_l1filenames, aps, **hiers):
        if len(spectrum_hdu.data) == 0:
            return
        replacements = column_name_acronym_replacements(colnames, 'rr')
        with unwind(safe_table, enumerated=True) as (row, nrow):
            specs, l1spectra, fibretargets = cls.read_l2product_table(this_fname, spectrum_hdu, row, nrow,
                                                          parent_l1filenames, IvarIngestedSpectrum,
                                                          IvarCombinedIngestedSpectrum,
                                                          ModelSpectrum, CombinedModelSpectrum,
                                                          True, None, 'RR', aps)
            redrock = cls.make_redrock_fit(specs, row, len(parent_l1filenames), replacements)
            l2 = cls.make_l2(l1spectra, nspec=len(parent_l1filenames), aps=aps, fibre_target=fibretargets[0], **hiers)
            l2.attach_optionals(redrock=redrock)
        l2, redrocks, *r = collect(l2, redrock, *specs)
        return l2, redrocks, FitSpecs(*r)

    @classmethod
    def read_rvspecfit(cls, this_fname, spectrum_hdu, colnames, safe_table: CypherVariable, parent_l1filenames, aps, **hiers):
        if len(spectrum_hdu.data) == 0:
            return
        replacements = column_name_acronym_replacements(colnames, 'rvs')
        with unwind(safe_table, enumerated=True) as (row, nrow):
            rvs_specs, l1spectra, fibretargets = cls.read_l2product_table(this_fname, spectrum_hdu, row, nrow,
                                                      parent_l1filenames, IngestedSpectrum,
                                                      CombinedIngestedSpectrum,
                                                      ModelSpectrum, CombinedModelSpectrum,
                                                      True, None, 'RVS', aps)
            rvspecfit = RVSpecfit(model_spectra=[rvs_specs.individual_models[i] for i in range(len(parent_l1filenames))],
                                  combined_model_spectrum=rvs_specs.combined_model, tables=row, tables_replace=replacements)
            l2 = cls.make_l2(l1spectra, nspec=len(parent_l1filenames), aps=aps, fibre_target=fibretargets[0], **hiers)
            l2.attach_optionals(rvspecfit=rvspecfit)
        l2, rvspecfits, *r = collect(l2, rvspecfit, *rvs_specs)
        return l2, rvspecfits, FitSpecs(*r)

    @classmethod
    def read_ferre(cls, this_fname, spectrum_hdu, colnames, safe_table: CypherVariable, parent_l1filenames, aps, **hiers):
        if len(spectrum_hdu.data) == 0:
            return
        replacements = column_name_acronym_replacements(colnames, 'fr')
        with unwind(safe_table, enumerated=True) as (row, nrow):
            ferre_specs, l1spectra, fibretargets = cls.read_l2product_table(this_fname, spectrum_hdu, row, nrow,
                                                        parent_l1filenames, IngestedSpectrum,
                                                        CombinedIngestedSpectrum,
                                                        ModelSpectrum, CombinedModelSpectrum,
                                                        True, None, 'FR', aps)
            ferre = Ferre(model_spectra=[ferre_specs.individual_models[i] for i in range(len(parent_l1filenames))],
                          combined_model_spectrum=ferre_specs.combined_model, tables=row, tables_replace=replacements)
            l2 = cls.make_l2(l1spectra, nspec=len(parent_l1filenames), aps=aps, fibre_target=fibretargets[0], **hiers)
            l2.attach_optionals(ferre=ferre)
        l2, ferres, *r = collect(l2, ferre, *ferre_specs)
        return l2, ferres, FitSpecs(*r)

    @classmethod
    def read_ppxf(cls, this_fname, spectrum_hdu, colnames, safe_table: CypherVariable, parent_l1filenames, aps, **hiers):
        if len(spectrum_hdu.data) == 0:
            return
        replacements = column_name_acronym_replacements(colnames, 'ppxf')
        with unwind(safe_table, enumerated=True) as (row, nrow):
            ppxf_specs, l1spectra, fibretargets = cls.read_l2product_table(this_fname, spectrum_hdu, row, nrow,
                                                         parent_l1filenames, None,
                                                         MaskedCombinedIngestedSpectrum,
                                                         None, CombinedModelSpectrum,
                                                         False, True, 'PPXF', aps)
            ppxf = PPXF(combined_model_spectrum=ppxf_specs.combined_model, tables=row, tables_replace=replacements)
            l2 = cls.make_l2(l1spectra, nspec=len(parent_l1filenames), aps=aps, fibre_target=fibretargets[0], **hiers)
            l2.attach_optionals(ppxf=ppxf)
        l2, ppxfs, *r = collect(l2, ppxf, *ppxf_specs)
        return l2, ppxfs, FitSpecs(*r)

    @classmethod
    def read_gandalf(cls, this_fname, spectrum_hdu, colnames, safe_table: CypherVariable, parent_l1filenames, aps, **hiers):
        if len(spectrum_hdu.data) == 0:
            return
        replacements = column_name_acronym_replacements(colnames, 'gand')
        with unwind(safe_table, enumerated=True) as (row, nrow):
            gandalf_specs, l1spectra, fibretargets = cls.read_l2product_table(this_fname, spectrum_hdu, row, nrow,
                                                          parent_l1filenames, None,
                                                         MaskedCombinedIngestedSpectrum,
                                                         None, GandalfModelSpectrum,
                                                         False, True, 'GAND', aps)
            gandalf, gandalf_extra_specs = cls.make_gandalf_structure(gandalf_specs, row, this_fname, nrow, replacements)
            l2 = cls.make_l2(l1spectra, nspec=len(parent_l1filenames), aps=aps, fibre_target=fibretargets[0], **hiers)
            l2.attach_optionals(gandalf=gandalf)
        l2, gandalfs, *r = collect(l2, gandalf, *gandalf_specs, *gandalf_extra_specs)
        return l2, gandalfs, FitSpecs(*r[:len(gandalf_specs)]), GandalfSpecs(*r[len(gandalf_specs):])

    @classmethod
    def add_zs_ids(cls, tbl: Table):
        for col in tbl.colnames:
            if 'CZZ_' in col and 'CHI2' not in col:
                tbl[f'{col}_START'] = tbl[col][:, 0]
                tbl[f'{col}_END'] = tbl[col][:, -1]
                tbl[f'{col}_STEP'] = tbl[col][:, 1] - tbl[col][:, 0]


    @classmethod
    def read(cls, directory: Union[Path, str], fname: Union[Path, str], slc: slice = None, part=None):
        if part is None:
            raise RuntimeError(f"{cls.__name__}.read() requires a part argument otherwise py2neo will crash")
        fname = Path(fname)
        directory = Path(directory)
        path = directory / fname
        header, aps = cls.read_header_and_aps(path)
        l1files = cls.parse_fname(header, fname)
        aps = APS(version=aps)
        hierarchies = cls.find_shared_hierarchy(path)
        astropy_hdus = fits.open(path)
        fnames = [l1.fname for l1 in l1files]
        assert len(fnames) > 1, f"{fname} has only one L1 file"
        safe_tables = {}
        safe_cypher_tables = {}
        for i, hdu in enumerate(astropy_hdus[1:4], 1):
            safe_tables[i] = filter_products_from_table(Table(hdu.data)[slc], MAX_REDSHIFT_GRID_LENGTH)
            cls.add_zs_ids(safe_tables[i])
            safe_cypher_tables[i] =  CypherData(safe_tables[i])
        if part == 'RVS':
            l2, specfits, specs = cls.read_rvspecfit(path, astropy_hdus[5], astropy_hdus[2].data.names,
                                                 safe_cypher_tables[2], fnames, aps, **hierarchies)
            hdu_node = 5
        elif part == 'FR':
            l2, specfits, specs = cls.read_ferre(path, astropy_hdus[5], astropy_hdus[2].data.names,
                                             safe_cypher_tables[2], fnames, aps, **hierarchies)
            hdu_node = 5
        elif part == 'PPXF':
            l2, specfits, specs = cls.read_ppxf(path, astropy_hdus[6], astropy_hdus[3].data.names,
                                            safe_cypher_tables[3], fnames, aps, **hierarchies)
            hdu_node = 6
        elif part == 'RR':
            l2, specfits, specs = cls.read_redrock(path, astropy_hdus[4], astropy_hdus[1].data.names,
                                               safe_cypher_tables[1], fnames, aps, **hierarchies)
            hdu_node = 4
        elif part == 'GAND':
            l2, specfits, specs, extra_specs = cls.read_gandalf(path, astropy_hdus[6], astropy_hdus[3].data.names,
                                                            safe_cypher_tables[3], fnames, aps, **hierarchies)
            hdu_node = 6
        else:
            raise ValueError(f"{part} is not a valid part")
        hdu_nodes, file, _ = cls.read_hdus(directory, fname, l2=l2, l1files=l1files, aps=aps, **hierarchies)
        hdu = hdu_nodes[hdu_node]
        names = {'logwvl': 'loglam', 'wvl': 'lambda'}
        if specs is not None:
            cls.attach_products_to_spectra(specs, part, hdu, names)
        if part == 'GAND':
            cls.attach_products_to_gandalf_extra_spectra(extra_specs, hdu)


class L2SingleFile(L2File):
    singular_name = 'l2single_file'
    children = []
    parents = [Multiple(L1SingleFile, 2, 3, constrain=(Exposure,)), APS, Multiple(L2Single)]
    L2 = L2Single
    L1 = L1SingleSpectrum


    @classmethod
    def find_shared_hierarchy(cls, path) -> Dict:
        header = cls.read_header_and_aps(path)[0]
        runids = cls.parser_ftypes_runids(header)[1]
        run = Run.find(id=runids[0])
        return {'exposure': Exposure.find(anonymous_children=[run])}


class L2StackFile(L2File):
    singular_name = 'l2stack_file'
    children = []
    parents = [Multiple(L1StackFile, 1, 3, constrain=(OB,)), APS, Multiple(L2Stack)]
    L2 = L2Stack
    L1 = L1StackSpectrum

    @classmethod
    def find_shared_hierarchy(cls, path) -> Dict:
        header = cls.read_header_and_aps(path)[0]
        return {'ob': OB.find(obid=header['OBID'])}


class L2SuperstackFile(L2File):
    singular_name = 'l2superstack_file'
    children = []
    parents = [Multiple(L1SingleFile, 0, 3, constrain=(OBSpec,)),
               Multiple(L1StackFile, 0, 3, constrain=(OBSpec,)),
               Multiple(L1SuperstackFile, 0, 3, constrain=(OBSpec,)), APS,
               Multiple(L2Superstack)]
    L2 = L2Superstack
    L1 = L1Spectrum

    @classmethod
    def find_shared_hierarchy(cls, path) -> Dict:
        header = cls.read_header_and_aps(path)[0]
        return {'obspec': OBSpec.find(xml=str(header['cat-name']))}


class L2SupertargetFile(L2File):
    singular_name = 'l2supertarget_file'
    match_pattern = 'WVE_*aps.fits'
    children = []
    parents = [Multiple(L1SupertargetFile, 2, 3, constrain=(WeaveTarget,)), APS, L2Supertarget]
    L2 = L2Supertarget
    L1 = L1SupertargetSpectrum

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