from language_model.graph_pb2 import Graph, FeatureNode, FeatureEdge
import argparse
from typing import Iterable, Tuple, Callable, Generator
import random
import networkx as nx

random.seed(2019)


def read_graph(file_path: str) -> Tuple[Graph, nx.DiGraph, Callable, Callable]:
    """
    Load a single data file, provide it and an accessible graph and edit function.

    Args:
        file_path: The path to a data file.
    """
    with open(file_path, "rb") as f:
        g = Graph()
        g.ParseFromString(f.read())
        gidx2gi = {n.id:i for i,n in enumerate(g.node)}
        def change_node_contents(node_i, new_content):
            g.node[gidx2gi[node_i]].contents = new_content
        def change_node_type(node_i, new_type):
            g.node[gidx2gi[node_i]].type = new_type
        node_map = {n.id: n for n in g.node}
        G = nx.DiGraph()
        nodes = [(n.id, {'data': n}) for n in g.node]
        G.add_nodes_from(nodes)
        G.add_edges_from([(e.sourceId,e.destinationId,{'data':e}) for e in g.edge])
        assert([] == [1 for n in G.nodes if 'data' not in G.nodes[n]])
        return G, g, change_node_contents, change_node_type

def write_graph(file_path: str, graph: Graph):
    with open(file_path, 'wb') as f:
        f.write(graph.SerializeToString())

def has_ancestor_in(G: nx.DiGraph, n: int, s: set):
    H = G.copy()
    for e in list(G.edges):
        if G.edges[e]['data'].type not in (FeatureEdge.AST_CHILD, FeatureEdge.ASSOCIATED_TOKEN):
            H.remove_edge(*e)
    ancestors = nx.ancestors(H, n)
    for a in ancestors:
        if G.nodes[a]['data'].contents in s:
            return True
    return False

#TODO Take care of references to imports... -> An idea might be to go over the difference in the content of the symbol cell
#TODO I should not remove TOKENS as IDS otherwise the filters would not work anymore...
def change_imports_to_token_type(G: nx.DiGraph, change_node_type: Callable):
    anc_set = {'IMPORT', 'IMPORTS', 'PACKAGE', 'PACKAGE_NAME'}
    for n in list(G.nodes):
        if has_ancestor_in(G, n, anc_set) and G.nodes[n]['data'].type == FeatureNode.IDENTIFIER_TOKEN:
            change_node_type(n, FeatureNode.TOKEN)

def find_variable_groups(G: nx.DiGraph) -> Generator:
    G = nx.Graph(G)
    for e in list(G.edges):
        if G.edges[e]['data'].type != FeatureEdge.LAST_USE:
            G.remove_edge(*e)
    for n in list(G.nodes):
        if G.nodes[n]['data'].type != FeatureNode.IDENTIFIER_TOKEN:
            G.remove_node(n)
    # now G is undirected and contains only LAST_USE edges and id nodes
    return nx.connected_components(G)



if __name__ == '__main__':
    G, g, change_name, change_type = read_graph('../Test.java.proto')
    change_imports_to_token_type(G, change_type)
