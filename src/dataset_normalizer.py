from language_model.graph_pb2 import Graph, FeatureNode, FeatureEdge
import argparse
from typing import Iterable, Tuple, Callable, Generator, Optional, List, Any, Dict, Set
import random
import networkx as nx
from enum import Enum
from glob import glob
from os.path import relpath, join, dirname
import os
from tqdm import tqdm
from multiprocessing import Pool
from collections import deque, defaultdict
# deterministic naming

random.seed(2019)

parser = argparse.ArgumentParser()
parser.add_argument("--source-path")
parser.add_argument("--target-path")
parser.add_argument("--types-in-names", action='store_true')
parser.add_argument("--max-postfix", default=3000)
args = parser.parse_args()

MAX_POSTFIX = args.max_postfix
TYPES_IN_NAMES = args.types_in_names

class IdTypes(Enum):
    LocalVariable = 1
    Class = 2
    Parameter = 3
    Argument = 4
    Function = 5

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


def write_graph(file_path: str, g: Graph):
    with open(file_path, 'wb') as f:
        f.write(g.SerializeToString())

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


# This is veeery slow, need to work on that: all ancestors are first computed, Graph is copied
def has_ancestor_in(G: nx.DiGraph, n: int, s: set, edges_to_use: set = {FeatureEdge.AST_CHILD, FeatureEdge.ASSOCIATED_TOKEN}):
    next_ancestors = deque([n])
    while next_ancestors:
        a = next_ancestors.pop()
        candids = [get_ancestor_with_edge(G, a, e) for e in edges_to_use]
        next_a = [n for n in candids if n is not None] # type: List[int]
        next_ancestors.extendleft(next_a)
        for n in next_a:
            contents = G.nodes[n]['data'].contents
            if contents in s:
                return contents
    return False

def type_of_newly_defined_variable(G: nx.DiGraph, usages: set):
    def output_name(var_type_name, java_type_name):
        if TYPES_IN_NAMES:
            return var_type_name+'-'+java_type_name
        else:
            return var_type_name
    for n in usages:
        anc_contents = has_ancestor_in(G, n, {'VARIABLE', 'METHOD', 'CLASS'}, edges_to_use={FeatureEdge.ASSOCIATED_TOKEN})
        if anc_contents == 'VARIABLE':
            anc_contents = has_ancestor_in(G, n, {'PARAMETERS', 'BLOCK', 'BODY', 'ANNOTATION'})
            if anc_contents == 'ANNOTATION':
                # Argument names in annotation are somehow handled as variables by the engine..
                return None
            var_node = get_ancestor_with_edge(G, n, FeatureEdge.ASSOCIATED_TOKEN)
            assert var_node is not None
            type_node = get_child_with_edge(G, var_node, FeatureEdge.AST_CHILD, ncontents='TYPE')
            if type_node is None:
                continue
            j_type = get_child_with_edge(G, type_node, FeatureEdge.AST_CHILD) or get_child_with_edge(G, type_node, FeatureEdge.ASSOCIATED_TOKEN)
            assert j_type is not None, 'failed for node: {}, type node id: {}'.format(G.nodes[n]['data'].contents, type_node)
            java_type = G.nodes[j_type]['data'].contents # type: str
            if java_type == 'PARAMETERIZED_TYPE':
                type_node = get_child_with_edge(G, j_type, FeatureEdge.AST_CHILD, ncontents='TYPE')
                assert type_node is not None
                j_type = get_child_with_edge(G, type_node, FeatureEdge.AST_CHILD) or get_child_with_edge(G, type_node, FeatureEdge.ASSOCIATED_TOKEN)
                assert j_type is not None
                java_type = G.nodes[j_type]['data'].contents
            if anc_contents == 'PARAMETERS':
                return output_name(IdTypes.Argument.name, java_type)
            elif anc_contents:
                return output_name(IdTypes.LocalVariable.name, java_type)
            else:
                var_par = get_ancestor_with_edge(G, var_node, FeatureEdge.AST_CHILD)
                if var_par:
                    members = G.nodes[var_par]['data']
                    if members.contents == 'MEMBERS':
                        return output_name(IdTypes.Parameter.name, java_type)
                raise ValueError('Could not map VARIABLE type id to any IdType for: ' + str(G.nodes[n]['data']))
            
        elif anc_contents == 'METHOD':
            return IdTypes.Function.name
        if anc_contents == 'CLASS':
            return IdTypes.Class.name
    return None

def get_ancestor_with_edge(G: nx.DiGraph, n: int, etype: FeatureEdge) -> Optional[int]:
    i_e = G.in_edges(n)
    for e in i_e:
        if G.edges[e]['data'].type == etype:
            return e[0]
    return None

def get_child_with_edge(G: nx.DiGraph, n: int, etype: FeatureEdge, ncontents: Optional[str] = None) -> Optional[int]:
    e = G.out_edges(n)
    for e in e:
        if G.edges[e]['data'].type == etype:
            if (ncontents is None) or (G.nodes[e[1]]['data'].contents == ncontents):
                return e[1]
    return None


def remove_all_import_package_ids(G: nx.DiGraph):
    anc_set = {'IMPORT', 'IMPORTS', 'PACKAGE', 'PACKAGE_NAME'}
    remove_nodes_with_property(G,
        lambda n: has_ancestor_in(G, n, anc_set) and G.nodes[n]['data'].type == FeatureNode.IDENTIFIER_TOKEN
    )

def remove_all_package_reference_ids(G: nx.DiGraph):
    identifier_nodes = [n for n in G.nodes if G.nodes[n]['data'].type == FeatureNode.IDENTIFIER_TOKEN]
    new_id = {idn: G.nodes[idn]['data'].contents == get_ancestor_with_edge(G, idn, FeatureEdge.ASSOCIATED_SYMBOL) for idn in identifier_nodes}
    remove_nodes_with_property(G, lambda n: not new_id[n])

def find_fun_ids_and_var_ids(G: nx.DiGraph) -> Tuple[List[int], List[int]]:
    ids = [i for i in G.nodes if G.nodes[i]['data'].type == FeatureNode.IDENTIFIER_TOKEN]
    fun_ids = [i for i in ids if has_ancestor_in(G, i, {'METHOD_INVOCATION', 'METHOD_SELECT', 'METHOD'}, edges_to_use={FeatureEdge.ASSOCIATED_TOKEN})]
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
    for c in comps:
        t = type_of_newly_defined_variable(G, set(c))
        if t:
            yield c, t

def normalize(G: nx.DiGraph, change_name: Callable):
    groups = find_id_groups(G)
    given_names = set() # type: Set[str]
    left_ids = defaultdict(lambda: list(range(1, MAX_POSTFIX + 1))) # type: Dict[str, List[int]]
    for group, type_name in groups:
        if len(left_ids[type_name]) == 0:
            raise ValueError('To small number range for current file')
        next_index = random.randint(0, len(left_ids[type_name])-1)
        proposed_name = type_name+'{:02d}'.format(left_ids[type_name][next_index])
        del left_ids[type_name][next_index]
        given_names.add(proposed_name)
        for i in group:
            change_name(i, proposed_name)

def load_edit_save(in_n_out):
    in_file, out_file = in_n_out
    G, g, change_name, change_type = read_graph(in_file)
    try:
        normalize(G, change_name)
    except Exception as v:
        raise ValueError(str(v) + ' with files: ' + str(in_n_out))
    write_graph(out_file, g)

def run_edit(source_dir, target_dir):
    in_files = glob(source_dir+'/**/*.proto', recursive=True)
    out_files = []
    for in_file in in_files:
        relative_path = relpath(in_file, source_dir)
        out_file = join(target_dir, relative_path)
        os.makedirs(dirname(out_file), exist_ok=True)
        out_files.append(out_file)
    pool = Pool(4)
    for _ in tqdm(pool.imap_unordered(load_edit_save, zip(in_files, out_files)), total=len(in_files)):
        pass

def print_as_text(G):
    tokens = [i for i in G.nodes if G.nodes[i]['data'].type in {FeatureNode.IDENTIFIER_TOKEN, FeatureNode.TOKEN}]
    tokens.sort(key=lambda i: G.nodes[i]['data'].startPosition)
    tokens_vs = [G.nodes[i]['data'].contents for i in tokens]
    new_liners = {'SEMI': ';', 'LBRACE': '{', 'RBRACE': '}'}
    tokens_vsn = [new_liners[t]+'\n' if t in new_liners else t for t in tokens_vs]
    print(' '.join(tokens_vsn))

if __name__ == '__main__':
    target_dir = args.target_path
    source_dir = args.source_path
    run_edit(source_dir, target_dir)
