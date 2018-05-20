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


def kruskal(graph, cliques_to_keep=1):
    for vertice in graph['vertices']:
        make_set(vertice)

    minimum_spanning_tree = set()
    edges = list(graph['edges'])
    edges.sort()

    clique_count = len(graph['vertices'])
    for edge in edges:
        weight, vertice1, vertice2 = edge
        if find(vertice1) != find(vertice2):
            union(vertice1, vertice2)
            minimum_spanning_tree.add(edge)

            # my stuff
            clique_count -= 1
            if clique_count == cliques_to_keep:
                cat = {}
                for _, v1, v2 in edges:
                    cat[v1] = find(v1)
                    cat[v2] = find(v2)

                # word counts
                word_counts = {}
                for c in list(cat.values()):
                    inc_dict(word_counts, c)

                # compact msg
                kruskal_cat2msg = {}
                for c, i in zip(word_counts.keys(), range(len(word_counts))):
                    kruskal_cat2msg[c] = i

                # now compute V
                V={}
                for v, c in cat.items():
                   V[v] = {'word': kruskal_cat2msg[c]}

                return V, word_counts




    #return sorted(minimum_spanning_tree)


def inc_dict(dict, key, increment=1):
    if key in dict.keys():
        dict[key] += increment
    else:
        dict[key] = increment


def main():



    n = wcs.color_dim()
    graph = {}
    graph['vertices'] = list(range(n))

    edges = set()
    for i in range(n-1):
        for j in range(i+1, n):
            edges.add((wcs.sim(i, j), i, j))
    graph['edges'] = edges

    for cliques_to_keep in range(3,12):
        V, word_counts = kruskal(graph, cliques_to_keep=cliques_to_keep)

        ccost = wcs.communication_cost_regier(V)
        minkcut_cost = wcs.min_k_cut_cost(V, cliques_to_keep)

        print('#words %d, min k-cut cost %f, reiger_cost %f' % (cliques_to_keep, minkcut_cost, ccost), end=" | ")
        print('word counts', word_counts.values())


if __name__ == "__main__":
    main()