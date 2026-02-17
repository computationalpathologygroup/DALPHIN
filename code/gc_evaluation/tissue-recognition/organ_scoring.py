import yaml
from collections import deque

class Node:
    def __init__(self, name, synonyms=None, parent=None):
        self.name = name
        self.synonyms = synonyms or []
        self.parent = parent
        self.children = []

    def __repr__(self):
        return f"Node({self.name})"


def load_taxonomy(taxonomy_file):
    with open(taxonomy_file, "r") as f:
        data = yaml.safe_load(f)

    nodes = _build_tree(data)
    lookup = _build_lookup(nodes)
    graph = _build_graph(nodes)
    return nodes, lookup, graph


def _build_tree(data, parent=None):
    nodes = []
    for key, value in data.items():
        synonyms = value.get('synonyms', [])
        node = Node(name=key, synonyms=synonyms, parent=parent)
        if parent:
            parent.children.append(node)
        nodes.append(node)
        if 'parts' in value:
            child_nodes = _build_tree(value['parts'], parent=node)
            nodes.extend(child_nodes)
    return nodes


def _build_lookup(nodes):
    lookup = {}
    for node in nodes:
        lookup[node.name.lower()] = node
        for syn in node.synonyms:
            lookup[syn.lower()] = node
    return lookup


def _build_graph(nodes):
    graph = {node: set() for node in nodes}
    for node in nodes:
        if node.parent:
            graph[node].add(node.parent)
            graph[node.parent].add(node)
            for sibling in node.parent.children:
                if sibling is not node:
                    graph[node].add(sibling)
                    graph[sibling].add(node)
    return graph


def compute_organ_score(input_term, ground_truth_term, lookup, graph):
    input_term = input_term.lower().strip()
    gt_terms = [t.strip().lower() for t in ground_truth_term.split(",")]

    return max(
        _hierarchical_score(input_term, gt, lookup, graph)
        for gt in gt_terms
    )


def _hierarchical_score(input_term, gt_term, lookup, graph):
    if input_term not in lookup or gt_term not in lookup:
        return 0.0

    if lookup[input_term] == lookup[gt_term]:
        return 1.0

    dist = _shortest_path(graph, lookup[gt_term], lookup[input_term])

    return {1: 0.75, 2: 0.5}.get(dist, 0.0)


def _shortest_path(graph, start, goal):
    queue = deque([(start, 0)])
    visited = {start}

    while queue:
        node, dist = queue.popleft()
        if node == goal:
            return dist
        for n in graph[node]:
            if n not in visited:
                visited.add(n)
                queue.append((n, dist + 1))
    return float("inf")
