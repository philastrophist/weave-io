import os
from pathlib import Path

HERE = Path(os.path.dirname(os.path.abspath(__file__)))

def read_procedure_text(cql_fname):
    with open(HERE / Path(cql_fname), 'r') as f:
        cypherlines = [l.strip() for l in f.readlines()]
    for i ,line in enumerate(cypherlines):
        if all(s == '/' for s in line):
            break
    else:
        raise ValueError('CQL header not found')
    header = cypherlines[:i]
    params = []
    returns = []
    requires = []
    description = 'no description given'
    for l in header:
        if '//' in l:
            l = l.split('//')[1].strip()
            if l.startswith('param:'):
                params.append([i.strip() for i in l[len('param:'):].split('=>')])
            if l.startswith('returns:'):
                returns.append([i.strip() for i in l[len('returns:'):].split('=>')])
            if l.startswith('description:'):
                description = l[len('description:'):].strip()
            if l.startswith('requires:'):
                requires = [x.strip() for x in l[len('requires:'):].strip().split(',')]
    query = '\n'.join(cypherlines[i+1:]).replace("'", "\\'")
    return query, returns, params, description, requires


def make_procedure(name, cql_fname, readwrite, tag=''):
    query, returns, params, description, requirements = read_procedure_text(cql_fname)
    for req in requirements:
        query = query.replace(req, f'custom.{req}{tag}')
    return f"CALL apoc.custom.asProcedure('{name}{tag}', '{query}', '{readwrite}', {returns}, {params}, '{description}')"


def get_all_procedures(mode, tag=''):
    procedures = []
    for f in HERE.glob('*.cql'):
        name = f.name.replace('.cql', '')
        procedures.append(make_procedure(name, str(f), mode, tag))
    return procedures
