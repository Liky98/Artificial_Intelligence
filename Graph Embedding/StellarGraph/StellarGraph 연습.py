import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from stellargraph import StellarGraph

square_edges = pd.DataFrame(
    {"source": ["a", "b", "c", "d", "a"], "target": ["b", "c", "d", "a", "c"]}
)
square_edges
#%%
square = StellarGraph(edges=square_edges)
print(square.info())
#%%
square_named = StellarGraph(
    edges=square_edges, node_type_default="corner", edge_type_default="line"
)
print(square_named.info())
#%%
square_edges_first_second = square_edges.rename(
    columns={"source": "first", "target": "second"}
)
square_edges_first_second
#%%
square_first_second = StellarGraph(
    edges=square_edges_first_second, source_column="first", target_column="second"
)
print(square_first_second.info())
#%%
square_node_data = pd.DataFrame(
    {"x": [1, 2, 3, 4], "y": [-0.2, 0.3, 0.0, -0.5]}, index=["a", "b", "c", "d"]
)
square_node_data
#%%
square_node_features = StellarGraph(square_node_data, square_edges)
print(square_node_features.info())