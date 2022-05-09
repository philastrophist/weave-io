import argparse
from weaveio import *
import os

def create_roles():



def query(args):
    data = Data(args.rootdir, args.host, args.port, args.dbname, args.password, args.username)
    raise NotImplementedError

def create_user(args):
    pass

if __name__ == '__main__':
    # create the top-level parser
    parser = argparse.ArgumentParser(description='WeaveIO management', add_help=False)
    parser.add_argument('--username', type=str, default=os.getenv('WEAVEIO_USER', 'weaveuser'))
    parser.add_argument('--password', type=str, default=os.getenv('WEAVEIO_PASSWORD', 'weavepassword'))

    # create sub-parser
    sub_parsers = parser.add_subparsers(help='sub-command help')

    # create the parser for the "ahoy" sub-command
    parser_query = sub_parsers.add_parser('query', help='execute a weaveio query')
    parser_query.add_argument('query', type=str, help='The query to execute')
    parser_query.add_argument('--rootdir', type=str, help='Root directory of the weaveio data')
    parser_query.add_argument('--host', type=str, help='Hostname of the weaveio database')
    parser_query.add_argument('--port', type=int, help='Port of the weaveio database')
    parser_query.add_argument('--dbname', type=str, help='Name of the weaveio database')
    parser_query.set_defaults(func=query)

    # create the parse for the "cool" sub-command
    parser_users = sub_parsers.add_parser('users', help='manage the database users')

    # create sub-parser for sub-command cool
    users_sub_parsers = parser_users.add_subparsers(help='sub-sub-command help')
    users_sub_parsers.required = True

    parser_user_create = users_sub_parsers.add_parser('create', help='create a new user')
    parser_user_create.set_defaults(func=create_user)

    parser_user_drop = users_sub_parsers.add_parser('drop', help='drop a user')


    args = parser.parse_args()
    args.func(args)