from weaveio.basequery.handler import Handler, Node, Path, Generator, Predicate, FullQuery, Branch

# g = Generator()
#
# run0 = g.node('Run')
# config0 = g.node('ArmConfig')
#
# p = Predicate(
#     Path(run0, '<--', config0),
#     exist_branches=[
#       Branch(Path(g.node('WeaveTarget', cname="WVE_02174085-0324395"), '-[*]->', run0)), '&',
#       Branch(Path(g.node('WeaveTarget', cname="WVE_02180982-0332433"), '-[*]->', run0)), '&',
#       Branch(Path(g.node('ArmConfig', camera='red'), '-->', run0)),
#     ],
#     return_properties=[(config0, 'vph')]
# )
# print(p.make_neo4j_statement(g))

Handler