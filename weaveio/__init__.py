import urllib

import json
from datetime import datetime

import logging

from .__version__ import __version__

import atexit
from typing import Callable

from boxing import boxing
from colored import fore, style
from pkg_info import get_pkg_info
from semver import compare


def parse_changes(description):
    log = description.split('machine-readable-change-log\n###########################\n')[-1]
    return log


def get_other_descriptions(name, version):
    return json.loads(urllib.request.urlopen(f'https://pypi.org/pypi/{name}/{version}/json', timeout=3).read())['info']['description']


class UpdateNotify(object):
    def __init__(self, name: str, version: str, **opts):
        self.name: str = name
        self.version: str = version
        self.pkg = get_pkg_info(self.name)
        self.is_updated: bool = self.is_latest_version()
        self.callback: Callable = opts.get('callback')
        self.message: str = opts.get('message')
        self.defer: bool = bool(opts.get('defer', False))

    def is_latest_version(self) -> bool:
        self.latest = self.pkg.version
        return True if compare(self.version, self.latest) >= 0 else False

    def render_changes(self):
        releases = [(k, datetime.strptime(v[0]['upload_time'], '%Y-%m-%dT%H:%M:%S')) for k, v in self.pkg.raw_data['releases'].items()]
        releases.sort(key=lambda x: x[1])
        release_names, _ = zip(*releases)
        changes = [parse_changes(get_other_descriptions(self.pkg.name, v)) for v in release_names]
        newer_changes = changes[release_names.index(self.pkg.version):]
        return '\n'.join(newer_changes), len(newer_changes)

    def notify(self) -> None:
        if self.is_updated:
            return
        if self.callback and callable(self.callback):
            action, arg = self.callback, None
        elif self.message:
            action, arg = print, self.message
        else:
            action, arg = print, self.default_message()
        if self.defer:
            atexit.register(action, arg) if arg else atexit.register(action)
        else:
            action(arg) if arg else action()

    def default_message(self) -> str:
        changes, nchanges = self.render_changes()
        version = fore.GREY_53 + self.version + style.RESET
        latest = fore.LIGHT_GREEN + self.latest + style.RESET
        command = fore.LIGHT_BLUE + 'pip install -U ' + self.name + style.RESET
        nchanges = fore.LIGHT_GREEN + str(nchanges) + style.RESET
        return boxing(f'Update available {version} → {latest} ({nchanges} new changes)\n' +
                      f'Run {command} to update\n'
                      f'{changes}')


try:
    UpdateNotify('weaveio', __version__).notify()
except Exception as e:
    logging.exception('There was a problem in alerting you to updated versions of the weaveio library...', exc_info=True)
