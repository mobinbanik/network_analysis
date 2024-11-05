# Graph Analysis and Visualization

This project performs graph analysis and visualization on a subset of the Twitch Gamers dataset. The dataset is sourced from [Stanford SNAP](https://snap.stanford.edu/data/twitch_gamers.html).

## Dataset

The Twitch Gamers dataset contains information about Twitch users and their interactions. For this project, we have used a subset of 10,000 edges from the dataset to perform our analysis.

## Project Structure

The project consists of the following main components:

1. **Data Loading and Preprocessing**
2. **Graph Metrics Computation**
3. **Closeness Centrality Computation**
4. **Graph Visualization**

## Data Loading and Preprocessing

We load the dataset using `pandas` and create a graph using `networkx`. We only use the first 10,000 edges from the dataset for our analysis.

```python
import pandas as pd
import networkx as nx

df = pd.read_csv('twitch_gamers/large_twitch_edges.csv', nrows=10000)
G = nx.from_pandas_edgelist(df, 'numeric_id_1', 'numeric_id_2')
```

## Graph Metrics Computation
We compute various graph metrics including:

- Number of Nodes
- Number of Edges
- Density
- Diameter
- Radius
- Average Degree
- Degree Variance
- Freeman's Degree Centralization
- Maximum Degree
- Size of Largest Weakly Connected Component
- Size of Largest Strongly Connected Component
- The results are saved in a CSV file and visualized using bar plots and histograms.

```python 
def compute_graph_metrics(G):
    # Compute metrics and save results
    ...
    results_df.to_csv('results/metrics/graph_metrics.csv')
    plt.savefig('results/metrics/graph_metrics.png')
    plt.savefig('results/metrics/degree_distribution.png')
```

## Closeness Centrality Computation
We compute the closeness centrality for all nodes in the graph and identify the top 3 nodes with the highest closeness centrality. The results are saved in a CSV file and visualized using histograms.

```python
def compute_closeness_centrality(G):
    # Compute closeness centrality and save results
    ...
    top_3_closeness.to_csv('results/closeness_centrality/top_3_closeness_centrality.csv', index=False)
    plt.savefig('results/closeness_centrality/closeness_centrality_distribution.png')
```

## Graph Visualization
We visualize the graph and save the visualization as an image.

```python
def visualize_graph(G):
    # Visualize the graph and save the image
    ...
    plt.savefig('results/visualization/graph_visualization.png')
```

## Results
The results of the analysis, including the computed metrics, top nodes by closeness centrality, and visualizations, are saved in the results directory. The directory structure is as follows:

```
results/
    metrics/
        graph_metrics.csv
        graph_metrics.png
        degree_distribution.png
    closeness_centrality/
        top_3_closeness_centrality.csv
        closeness_centrality_distribution.png
    visualization/
        graph_visualization.png
```

## Conclusion
This project demonstrates how to perform basic graph analysis and visualization using Python libraries such as pandas, networkx, matplotlib, and seaborn. The analysis provides insights into the structure and properties of the Twitch Gamers network

## References
Twitch Gamers Dataset - Stanford SNAP : https://snap.stanford.edu/data/twitch_gamers.html
