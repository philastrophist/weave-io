import os
from pathlib import Path

HERE = Path(os.path.dirname(os.path.abspath(__file__)))

def make_procedure(name, cql_fname, readwrite):
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
    description = 'no description given'
    for l in header:
        if '//' in l:
            print(l)
            l = l.split('//')[1].strip()
            if l.startswith('param:'):
                params.append([i.strip() for i in l[len('param:'):].split('=>')])
            if l.startswith('returns:'):
                returns.append([i.strip() for i in l[len('returns:'):].split('=>')])
            if l.startswith('description:'):
                description = l[len('description:'):].strip()
    query = '\n'.join(cypherlines[i+1:]).replace("'", "\\'")
    return f"CALL apoc.custom.asProcedure('{name}', '{query}', '{readwrite}', {returns}, {params}, '{description}')"


if __name__ == '__main__':
    p = make_procedure('multimerge', 'multimerge.cql', 'write')
    print(p)