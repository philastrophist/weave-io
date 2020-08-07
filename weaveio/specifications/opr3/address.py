
LEVEL = ['raw', 'l1', 'l2']
RESOLUTION = ['L', 'H']
ARM = ['R', 'B']
MODE = ['MOS', 'LIFU', 'mLIFU']


class Address:
    coordinates = ['level', 'obid', 'obname', 'cname', 'runid', 'surveyname', 'semester', 'arm', 'resolution', 'binning', 'mode']

    def __init__(self, level=None, obid=None, obname=None, cname=None, runid=None,
                 surveyname=None, semester=None, arm=None, resolution=None, binning=None, mode=None):
        self.level = self.check(level, str, choices=LEVEL)
        self.obid = self.check(obid, str)
        self.obname = self.check(obname, str)
        self.cname = self.check(cname, str)
        self.runid = self.check(runid, str, validator=lambda r: r.startswith('r'), failmsg='runid must start with "r"')
        self.surveyname = self.check(surveyname, str)
        self.semester = self.check(semester, str, validator=lambda s: s.startswith('S'), failmsg='semester must start with "S"')
        self.arm = self.check(arm, str, choices=ARM)
        self.resolution = self.check(resolution, str, choices=RESOLUTION)
        self.binning = self.check(binning, int)
        self.mode = self.check(mode, str, choices=MODE)
        super().__init__()
    