import weaveio
import subprocess

def version(args):
    print(f"weaveio version: {weaveio.__version__}")

def console(args):
    import IPython
    from traitlets.config import Config
    c = Config()
    c.InteractiveShellApp.exec_lines = [
        "from weaveio import *",
        "data = Data()",
    ]
    IPython.start_ipython(config=c, user_ns=args.__dict__, matplotlib='auto', colors='Linux',
                          header=f'WeaveIO console version {weaveio.__version__}')

def jupyter(args):
    subprocess.run(['conda', 'activate', 'weaveio', '&&',
                             'jupyter', 'notebook', '--no-browser', '--ip', args.ip,
                             '--port', args.port, '--allow-root'])

def upgrade(args):
    subprocess.run(['conda', 'activate', 'weaveio', '&&', 'pip', 'install', '--upgrade', 'weaveio'])

def add_data(args):
    raise NotImplementedError()

def main():
    from argparse import ArgumentParser
    parser = ArgumentParser()
    sub_parsers = parser.add_subparsers(help='sub-command help')

    parser_version = sub_parsers.add_parser('version', help='display version')
    parser_version.set_defaults(func=version)
    parser_console = sub_parsers.add_parser('console', help='start weaveio ipython console')
    parser_console.set_defaults(func=console)
    parser_jupyter = sub_parsers.add_parser('jupyter', help='start a jupyter notebook in the weaveio environment in this directory')
    parser_jupyter.set_defaults(func=jupyter)
    parser_upgrade = sub_parsers.add_parser('upgrade', help='upgrade weaveio through pip')
    parser_upgrade.set_defaults(func=upgrade)
    parser_add_data = sub_parsers.add_parser('add_data', help='add data to weaveio')
    parser_add_data.set_defaults(func=add_data)


    args = parser.parse_args()
    args.func(args)

if __name__ == '__main__':
    main()