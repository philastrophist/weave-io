from functools import reduce
from pathlib import Path
from typing import Union, Any, Dict, Set, List

import networkx as nx

from weaveio.graph import Graph
from weaveio.hierarchy import File, Raw, L1Single, L1Stack, L1SuperStack, L1SuperTarget, L2Single, L2Stack, L2SuperTarget


class Address(dict):
    pass


class Data:
    filetypes = []

    def __init__(self, rootdir: Union[Path, str]):
        super(Data, self).__init__()
        self.graph = Graph()
        self.filelists = {}
        self.rootdir = Path(rootdir)
        self.read_in_directory()
        self.address = Address()

    def read_in_directory(self):
        for filetype in self.filetypes:
            self.filelists[filetype] = filetype.match(self.rootdir)
        with self.graph:
            for filetype, files in self.filelists.items():
                for file in files:
                    filetype(file).read()
            self.graph.remove_node('Factor')

    def _match_files(self, factor_name: str, factor_value: Any, filetype: str) -> Set[str]:
        """
        Match files by building a list of shortest paths that have the subpath:
            (factor_name)->(factor_value)->...->(file)->(filetype)
        return [(file) from each valid path]
        """
        paths, lengths = zip(*[(p, len(p)) for p in nx.all_shortest_paths(self.graph, factor_name, filetype)])
        node = f'{factor_name}({factor_value})'
        paths = [p for p in paths if len(p) == min(lengths) and node in p]
        files = {p[-2] for p in paths}
        return files

    def _query_files(self, filetype: str, factors: Dict[str, Any]) -> Set[str]:
        """
        Match all valid files
        Do this for each given factor and take then intersection of the file set as the result
        """
        return reduce(lambda a, b: a & b, [self._match_files(k, v, filetype) for k, v in factors.items()])

    def graph_view(self, include: Union[str, List[str]] = None,
                   exclude: Union[str, List[str]] = None) -> nx.Graph:
        """
        Returns a view (not a copy) into the data graph but hiding any nodes in `exclude` and any
        nodes that don't have a path to any node in `include`.
        :param include: Nodes that all resulting graph nodes should have a path to
        :param exclude: Nodes that should be included
        :return: networkx.Graph
        """
        if isinstance(include, str):
            include = [include]
        elif include is None:
            include = []
        if isinstance(exclude, str):
            exclude = [exclude]
        elif exclude is None:
            exclude = []
        view = self.graph
        if include:
            view = nx.subgraph_view(view, lambda n: any(nx.has_path(self.graph, n, i) for i in include))
        if exclude:
            view = nx.subgraph_view(view, lambda n: not any(n.startswith(i) for i in exclude))
        isolates = nx.isolates(view)
        view = nx.subgraph_view(view, lambda n: n not in isolates)
        return view


class HomogeneousStore:
    pass


class OurData(Data):
    filetypes = [Raw, L1Single, L1Stack, L1SuperStack, L1SuperTarget, L2Single, L2Stack, L2SuperTarget]
