#!/usr/bin/env python

import inspect
import os
import subprocess
import sys
from textwrap import dedent

try:
    import weaveio
    from weaveio import *
except ImportError as e:
    raise Exception(f"weaveio is not installed. Please install it and activate its environment") from e


def shell_command(command, capture_output=True, echo=True, dryrun=False):
    if echo:
        print(f"running `{command}`")
    if dryrun:
        return
    if capture_output:
        shell_command = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
        stdout, stderr = shell_command.communicate()
        shell_command.wait(5)
        if shell_command.returncode != 0:
            raise Exception(f"{command} failed with return code {shell_command.returncode}")
        return stdout.decode('utf-8').strip('\n')
    return subprocess.Popen(command, shell=True)


def get_default_args(func):
    signature = inspect.signature(func)
    return {
        k: v.default
        for k, v in signature.parameters.items()
        if v.default is not inspect.Parameter.empty
    }


def get_neo4j_version():
    return shell_command('echo "call dbms.components() yield name, versions, edition unwind versions as version return name, version, edition;" | cypher-shell -u weaveuser -p weavepassword | tail -n 1')


def get_apoc_version():
    return shell_command('echo "return apoc.version() AS output;" | cypher-shell -u weaveuser -p weavepassword | tail -n 1')


def get_python_version():
    import sys
    return f"Python version: {sys.version}"


def version(args):
    print(f"python version: {get_python_version()}")
    print(f"weaveio version: {weaveio.__version__}")
    print(f"neo4j version: {get_neo4j_version()}")
    print(f"apoc version: {get_apoc_version()}")


def console(args):
    import IPython
    data = Data(rootdir=args.rootdir, host=args.host, port=args.port, user=args.user, password=args.password, dbname=args.dbname)
    header = dedent(f"""WeaveIO console version {weaveio.__version__}
data = Data(rootdir='{data.rootdir}', host='{data.host}', port='{data.port}', user='{data.user}', dbname='{data.dbname}')
run `%matplotlib` to allow interactive plots""")
    IPython.embed(header=header, matplotlib='auto', colors='Linux')

def jupyter_start(args):
    print('Remember to forward the port to the notebook server')
    if args.port is None:
        shell_command('jupyter notebook --no-browser --ip 0.0.0.0 --allow-root', capture_output=False, dryrun=args.dryrun)
    else:
        shell_command(f'jupyter notebook --no-browser --ip 0.0.0.0 --port {args.port} --allow-root', capture_output=False, dryrun=args.dryrun)
    print('Notebook server started')
    print('To stop the notebook server, run weaveio jupyter --stop')


def jupyter_stop(args):
    if args.port is None:
        print('killing all weaveio jupyter servers at the following ports')
        print(shell_command("pgrep -f 'weaveio/bin/jupyter-notebook'", capture_output=True, echo=False, dryrun=args.dryrun))
        input('press enter to continue\n>>>')
        shell_command("pkill -f 'weaveio/bin/jupyter-notebook'", capture_output=False, dryrun=args.dryrun)


def jupyter(args):
    if args.list:
        print('weaveio jupyter servers at the following ports:')
        try:
            print(shell_command("pgrep -f 'weaveio/bin/jupyter-notebook'", capture_output=True, echo=False, dryrun=args.dryrun))
        except Exception:
            pass
    elif args.stop:
        return jupyter_stop(args)
    else:
        return jupyter_start(args)


def upgrade(args):
    return shell_command(f"{str(Path(sys.executable).parents[0] / 'pip install -U weaveio --no-cache-dir')}", capture_output=False, dryrun=args.dryrun)

def add_data(args):
    raise NotImplementedError(f"This feature has not yet been written")

def main():
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--dryrun', action='store_true', help='do not actually run any commands')
    sub_parsers = parser.add_subparsers(help='sub-command help')

    parser_version = sub_parsers.add_parser('version', help='display weaveio version')
    parser_version.set_defaults(func=version)
    parser_console = sub_parsers.add_parser('console', help='start weaveio ipython console')
    parser_console.add_argument('--dbname', default='production', help='root directory of data')
    parser_console.add_argument('--rootdir', default=None, help='root directory of data')
    parser_console.add_argument('--host', default=None, help='neo4j host')
    parser_console.add_argument('--port', default=None, help='neo4j port')
    parser_console.add_argument('--user', default=None, help='neo4j user')
    parser_console.add_argument('--password', default=None, help='neo4j password')
    parser_console.set_defaults(func=console)
    parser_jupyter = sub_parsers.add_parser('jupyter', help='start a jupyter notebook in the weaveio environment in this directory')
    parser_jupyter.add_argument('--port', help='port to bind to, default is port found in jupyter configuration')
    parser_jupyter.add_argument('--stop', action='store_true', help='stop weaveio jupyter server at selected port or all of them if not specified')
    parser_jupyter.add_argument('--list', action='store_true', help='list weaveio jupyter servers')
    parser_jupyter.set_defaults(func=jupyter)
    parser_upgrade = sub_parsers.add_parser('upgrade', help='upgrade weaveio through pip')
    parser_upgrade.set_defaults(func=upgrade)
    parser_add_data = sub_parsers.add_parser('add_data', help='add data to weaveio')
    parser_add_data.set_defaults(func=add_data)

    args = parser.parse_args()
    if args.dryrun:
        print(f"dryrun mode: {args}")
    if 'func' not in args:
        parser.print_help()
        exit(1)
    args.func(args)

if __name__ == '__main__':
    main()