import collections

edges = collections.defaultdict(dict)

def add_edge(u, v, w):
    if v not in edges[u] or edges[u][v] > w:
        edges[u][v] = w
    if u not in edges[v] or edges[v][u] > w:
        edges[v][u] = w

# Generate base dense connections
for i in range(100):
    if i + 1 < 100: add_edge(i, i+1, 10)
    if i + 2 < 100: add_edge(i, i+2, 15)
    if i + 3 < 100: add_edge(i, i+3, 20)
    if i + 4 < 100: add_edge(i, i+4, 25)
    if i + 10 < 100: add_edge(i, i+10, 50)

# The trap paths (looks like big jumps, but costs too much)
add_edge(0, 50, 100)
add_edge(50, 99, 100)

# The actual shortest path (Distance = 8)
add_edge(0, 27, 2)
add_edge(27, 54, 2)
add_edge(54, 81, 2)
add_edge(81, 99, 2)

# Output the exact prompt format
print("📝 QUESTION:")
for i in range(100):
    connections = []
    for neighbor, weight in sorted(edges[i].items()):
        connections.append(f"{neighbor} (weight: {weight})")
    print(f"Node {i} is connected to nodes {', '.join(connections)}.")
print("\nQuestion: Calculate the distance of the shortest path from node 0 to node 99.")