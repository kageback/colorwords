import wcs

parent = dict()
rank = dict()


def make_set(vertice):
    parent[vertice] = vertice
    rank[vertice] = 0


def find(vertice):
    if parent[vertice] != vertice:
        parent[vertice] = find(parent[vertice])
    return parent[vertice]


def union(vertice1, vertice2):
    root1 = find(vertice1)
    root2 = find(vertice2)
    if root1 != root2:
        if rank[root1] > rank[root2]:
            parent[root2] = root1
        else:
            parent[root1] = root2
        if rank[root1] == rank[root2]: rank[root2] += 1


def kruskal(graph):
    for vertice in graph['vertices']:
        make_set(vertice)
        minimum_spanning_tree = set()
        edges = list(graph['edges'])
        edges.sort()
    # print edges
    for edge in edges:
        weight, vertice1, vertice2 = edge
        if find(vertice1) != find(vertice2):
            union(vertice1, vertice2)
            minimum_spanning_tree.add(edge)

    return sorted(minimum_spanning_tree)


def main():
    n = wcs.color_dim()
    graph = {}
    graph['vertices'] = list(range(n))

    edges = set()
    for i in range(n-1):
        for j in range(i+1, n):
            edges.add((wcs.sim(i, j), i, j))
    graph['edges'] = edges

    print(kruskal(graph))

    #E_len = n*(n-1)/2

    # print('Graph built!')
    #
    # E_sorted_list = sorted(E.items(), key=operator.itemgetter(1))
    # E_sorted_list.reverse()
    #
    # while len(C) >= 11:
    #     e = E_sorted_list.pop()


if __name__ == "__main__":
    main()