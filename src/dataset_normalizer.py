from language_model.graph_pb2 import Graph, FeatureNode, FeatureEdge
import argparse
from typing import Iterable, Tuple, Callable, Generator, Optional, List, Any, Dict
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

def invert_dict(d: dict) -> Dict[Any, List[Any]]:
    inv_map = {} # type: Dict[Any, List[Any]]
    for k, v in d.items():
        inv_map[v] = inv_map.get(v, [])
        inv_map[v].append(k)
    return inv_map

def remove_edges_with_property(G: nx.Graph, rm_prop: Callable):
    for e in list(G.edges):
        if rm_prop(e):
            G.remove_edge(*e)

def remove_nodes_with_property(G: nx.Graph, rm_prop: Callable):
    for n in list(G.nodes):
        if rm_prop(n):
            G.remove_node(n)


def has_ancestor_in(G: nx.DiGraph, n: int, s: set):
    H = G.copy()
    remove_edges_with_property(H,
        lambda e: G.edges[e]['data'].type not in (FeatureEdge.AST_CHILD, FeatureEdge.ASSOCIATED_TOKEN)
    )
    ancestors = nx.ancestors(H, n)
    for a in ancestors:
        if G.nodes[a]['data'].contents in s:
            return True
    return False

def is_newly_defined_variable(G: nx.DiGraph, usages: set):
    H = G.copy()
    remove_edges_with_property(H, lambda e: G.edges[e]['data'].type not in {FeatureEdge.ASSOCIATED_TOKEN})
    for n in usages:
        if has_ancestor_in(H, n, {'VARIABLE', 'METHOD', 'CLASS'}):
            return True
    return False

def get_ancestor_with_edge(G: nx.DiGraph, n: int, etype: FeatureEdge) -> Optional[int]:
    i_e = G.in_edges(n)
    for e in i_e:
        if G.edges[e]['data'].type == etype:
            return e[0]
    return None

#TODO Take care of references to imports... -> An idea might be to go over the difference in the content of the symbol cell
#TODO I should not remove TOKENS as IDS otherwise the filters would not work anymore...
def remove_all_import_package_ids(G: nx.DiGraph):
    anc_set = {'IMPORT', 'IMPORTS', 'PACKAGE', 'PACKAGE_NAME'}
    remove_nodes_with_property(G,
        lambda n: has_ancestor_in(G, n, anc_set) and G.nodes[n]['data'].type == FeatureNode.IDENTIFIER_TOKEN
    )

def remove_all_package_reference_ids(G: nx.DiGraph):
    H = G.copy()
    remove_nodes_with_property(H, lambda n: G.nodes[n]['data'].type not in (FeatureNode.IDENTIFIER_TOKEN, FeatureNode.SYMBOL_TYP))
    
    # build a dictionary of identifier ids to their contents + their symbol_typs contents
    # conpare these contents to see if it is a package reference
    # What about references that are not explicitely imported? Remove everything beginning in {'java',...}?
    identifier_nodes = [n for n in H.nodes if G.nodes[n]['data'].type == FeatureNode.IDENTIFIER_TOKEN]
    new_id = {idn: G.nodes[idn]['data'].contents == G.nodes[G.in_edges(idn)[0]].contents for idn in identifier_nodes}
    remove_nodes_with_property(G, lambda n: not new_id[n])

def find_fun_ids_and_var_ids(G: nx.DiGraph) -> Tuple[List[int], List[int]]:
    ids = [i for i in G.nodes if G.nodes[i]['data'].type == FeatureNode.IDENTIFIER_TOKEN]
    H = G.copy()
    remove_edges_with_property(H, lambda e: G.edges[e]['data'].type not in {FeatureEdge.ASSOCIATED_TOKEN})
    fun_ids = [i for i in ids if has_ancestor_in(H, i, {'METHOD_INVOCATION', 'METHOD_SELECT', 'METHOD'})]
    return fun_ids, list(set(ids)-set(fun_ids))


def find_function_usage_groups(G: nx.DiGraph, fun_ids: List[int]) -> List[List[int]]:
    associated_symbols = {i: get_ancestor_with_edge(G,i,FeatureEdge.ASSOCIATED_SYMBOL) or \
                             get_ancestor_with_edge(
                                 G, get_ancestor_with_edge(G,i,FeatureEdge.ASSOCIATED_TOKEN), FeatureEdge.ASSOCIATED_SYMBOL
                             )
                          for i in fun_ids}
    associated_symbols = {k: G.nodes[v]['data'].contents for k,v in associated_symbols.items()}
    assert(all(associated_symbols.values()))
    symbols_to_ids = invert_dict(associated_symbols)
    return list(symbols_to_ids.values())

def find_variable_usage_groups(G: nx.DiGraph, var_ids: List[int]) -> List[List[int]]: 
    G = nx.Graph(G)
    remove_edges_with_property(G, lambda e: G.edges[e]['data'].type not in (FeatureEdge.LAST_USE, FeatureEdge.LAST_LEXICAL_USE, FeatureEdge.LAST_WRITE))
    remove_nodes_with_property(G, lambda n: n not in var_ids)
    # now G is undirected and contains only LAST_USE edges and id nodes
    comps = nx.connected_components(G)
    return comps


def find_id_groups(G: nx.DiGraph) -> Generator:
    G = G.copy()
    remove_all_import_package_ids(G)
    fun_ids, var_ids = find_fun_ids_and_var_ids(G)
    fun_groups = find_function_usage_groups(G, fun_ids)
    var_groups = find_variable_usage_groups(G, var_ids)
    comps = set(frozenset(s) for s in fun_groups).union(set(frozenset(s) for s in var_groups))
    flat_comps = [e for s in comps for e in s]
    assert(len(flat_comps) == len(set(flat_comps)))
    # validate all comps are valid: have one variable dependent
    return (c for c in comps if is_newly_defined_variable(G, set(c)))



if __name__ == '__main__':
    G, g, change_name, change_type = read_graph('../Test.java.proto')
