from graphviz import Digraph

def topological_sort(start_node):
    visited = set()
    res = []

    def _dfs(node):
        if node not in visited:
            visited.add(node)
            if hasattr(node, 'input_nodes'):
                for input_node in node.input_nodes:
                    _dfs(input_node)
            res.append(node)
            
    _dfs(start_node)
    return res

def to_graphviz(node):
    graph = topological_sort(node)
    
    f = Digraph()
    f.attr(rankdir='LR', size='100,100')
    f.attr('node', shape='circle')

    for node in graph:
        f.node(node.name, label=node.name.split('/')[0], shape='circle')
    for node in graph:
        if hasattr(node, 'input_nodes'):
            for e in node.input_nodes:
                f.edge(e.name, node.name, label=e.name)
    return f