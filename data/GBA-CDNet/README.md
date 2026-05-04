GBA-CDNet (Greater Bay Area Cost-Distance Network Dataset)

GBA-CDNet is a real-world, large-scale network design dataset for Cost-Distance problems
and low-altitude drone logistics planning. It is built from publicly available geographic
information and logistics-related statistics in the Guangdong-Hong Kong-Macao Greater Bay
Area. Nodes represent logistics-related locations (e.g., distribution sites, charging
points, or transfer hubs). Edges represent feasible drone flight corridors. Each edge has
two weights: construction cost and routing distance, capturing the tradeoff between
infrastructure cost and transport efficiency.

Each instance is a rooted weighted graph. The root node is the central depot or hub.
Some nodes are sources with non-negative demand weights, indicating logistics demand.
The task is to select a connected subgraph minimizing construction cost plus weighted
shortest-path distance to the root.

The full region is partitioned into 9 subgraphs by assigning each node to its nearest
core center (based on direct geographic distance to the centers). The resulting instances
are named GBA-CDNet-1 to GBA-CDNet-9 and vary in size, density, and source count.

Files
Each subgraph lives in its own folder:
- GBA-CDNet-1 ... GBA-CDNet-9
  - nodes.csv
    - node: node index (0-based, int)
    - weight: demand weight for this node (float, >= 0)
    - type: node type (int)
      - 0: root (center) node
      - 1: source node (weight > 0)
      - 2: Steiner/relay node (weight == 0)
  - edges.csv
    - u, v: endpoints (0-based, undirected edge)
    - construction: construction cost for (u, v)
    - transport: routing distance (already scaled if a scale is used)

Notes
- Edges are undirected; loaders symmetrize into both (u, v) and (v, u).
