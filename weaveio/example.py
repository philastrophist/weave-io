from .writequery import CypherData, unwind, collect, groupby


def get_fibretargets():
    srvyinfo = CypherData(df_svryinfo)
    fibinfo = CypherData(df_fibinfo)
    with unwind(srvyinfo) as svryrow:
        with unwind(svryrow['surveyname']) as surveyname:
            survey = Survey(surveyname=surveyname)
        surveys = collect(survey)
        prog = SubProgramme(targprog=svryrow['targprog'], surveys=surveys)
        cat = SurveyCatalogue(catname=svryrow['targcat'], subprogramme=prog)
    cat_collection = collect(cat)
    cats = groupby(cat_collection, 'catname')
    with unwind(fibinfo) as fibrow:
        cat = cats[fibrow['targcat']]
        weavetarget = WeaveTarget(cname=fibrow['cname'])
        surveytarget = SurveyTarget(surveycatalogue=cat, weavetarget=weavetarget, tables=fibrow)
        photometry = TargetPhotometry(surveytarget=surveytarget, tables=fibrow)
        fibre = Fibre(fibreid=fibrow['fibreid'])
        fibtarget = FibreTarget(surveytarget=surveytarget, fibre=fibre, targetphotometry=photometry)
    fibretargets = collect(fibtarget)


def add_raw():
    fibretargets = get_fibretargets()
    fibreset = FibreSet(fibretargets=fibretargets)
    obspec = OBSpec(xml=xml, obtitle=obtitle, obstemp=obstemp, fibreset=fibreset, progtemp=progtemp)
    ob = OB(obid=obid, obstartmjd=mjd, obspec=obspec)
    exposure = Exposure(expmjd=expmjd, ob=ob)
    run = Run(runid=runid, exposure=exposure, armconfig=armconfig)
    system = System(sysver)
    simulator = Simulator(simver)
    casu = CASU(casuver)
    raw = RawSpectrum(run=run, casu=casu, simulator=simulator, system=system)
    rawfile = RawFile(fname=fname, raw=raw)


def add_single():
    system = System(sysver)
    simulator = Simulator(simver)
    casu = CASU(casuver)
    run = Run(runid=runid, exposure=None)  # implies matching not creation
    raw = RawSpectrum(run=run, casu=casu, simulator=simulator, system=system)
    fibretargets = get_fibretargets()
    with unwind(fibretargets, checksums) as (fibretarget, checksum):
        spec = L1SingleSpectrum(rawspectrum=raw, fibretarget=fibretarget, casu=casu, checksum=checksum)
    specs = collect(spec)
    single = L1SingleFile(fname=fname, l1singlespectra=specs)


def add_stack():
    system = System(sysver)
    simulator = Simulator(simver)
    casu = CASU(casuver)
    fibretargets = get_fibretargets()
    with unwind(checksums, fibretargets) as (checksum, fibretarget):
        with unwind(runids) as runid:
            run = Run(runid=runid, exposure=None)  # implies matching not creation
            raw = RawSpectrum(run=run, casu=casu, simulator=simulator, system=system)
            spec = L1SingleSpectrum(rawspectrum=raw, fibretarget=fibretarget, casu=casu, checksum=None)
        specs = collect(spec)
        stack = L1StackSpectrum(l1singlespectra=specs, casu=casu, checksum=checksum)
    stacks = collect(stack)
    L1StackFile(fname=fname, l1stackspectra=stacks)


