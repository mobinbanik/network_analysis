import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
import os


def visualize_graph(G):
    """
    Visualizes the graph and saves the visualization as an image.

    Parameters:
    G (networkx.Graph): The graph to visualize.
    """
    # Create results directory if it doesn't exist
    os.makedirs('results/visualization', exist_ok=True)
    
    # Draw the graph
    plt.figure(figsize=(10, 10))
    nx.draw(G, node_size=10)
    plt.savefig('results/visualization/graph_visualization.png')


def compute_graph_metrics(G):
    """
    Computes various graph metrics and saves the results as a CSV file and plots.

    Parameters:
    G (networkx.Graph): The graph to compute metrics for.
    """
    # Create results directory if it doesn't exist
    os.makedirs('results/metrics', exist_ok=True)
    
    # Compute the number of nodes and edges
    num_nodes = G.number_of_nodes()
    num_edges = G.number_of_edges()

    # Compute graph density
    density = nx.density(G)

    # Compute degrees
    degrees = [d for n, d in G.degree()]
    avg_degree = sum(degrees) / num_nodes
    degree_variance = pd.Series(degrees).var()
    max_degree = max(degrees)

    # Compute Freeman's degree centralization
    max_deg = max_degree
    centralization_num = sum(max_deg - d for d in degrees)
    centralization_den = (num_nodes - 1) * (num_nodes - 2)
    degree_centralization = centralization_num / centralization_den if centralization_den != 0 else 0

    # Compute connected components
    connected_components = list(nx.connected_components(G))
    largest_cc = max(connected_components, key=len)
    largest_wcc_size = len(largest_cc)

    # Compute diameter and radius
    if nx.is_connected(G):
        diameter = nx.diameter(G)
        radius = nx.radius(G)
    else:
        G_lcc = G.subgraph(largest_cc)
        diameter = nx.diameter(G_lcc)
        radius = nx.radius(G_lcc)

    # For undirected graphs, strongly connected components are not applicable
    largest_scc_size = 0  # Replace "N/A" with 0

    # Create a DataFrame to display the results
    metrics = {
        'Number of Nodes': num_nodes,
        'Number of Edges': num_edges,
        'Density': density,
        'Diameter': diameter,
        'Radius': radius,
        'Average Degree': avg_degree,
        'Degree Variance': degree_variance,
        "Freeman's Degree Centralization": degree_centralization,
        'Maximum Degree': max_degree,
        'Size of Largest Weakly Connected Component': largest_wcc_size,
        'Size of Largest Strongly Connected Component': largest_scc_size
    }
    results_df = pd.DataFrame.from_dict(metrics, orient='index', columns=['Value'])

    # Print the results table
    print(results_df)

    # Save the results to a CSV file
    results_df.to_csv('results/metrics/graph_metrics.csv')

    # Plotting some metrics
    plt.figure(figsize=(10, 6))
    sns.barplot(x=results_df.index, y=results_df['Value'])
    plt.xticks(rotation=90)
    plt.title('Graph Metrics')
    plt.savefig('results/metrics/graph_metrics.png')

    # Plot degree distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(degrees, bins=30, kde=True)
    plt.title('Degree Distribution')
    plt.xlabel('Degree')
    plt.ylabel('Frequency')
    plt.savefig('results/metrics/degree_distribution.png')


def compute_closeness_centrality(G):
    """
    Computes the closeness centrality for all nodes and saves the top 3 nodes and the distribution plot.

    Parameters:
    G (networkx.Graph): The graph to compute closeness centrality for.
    """
    # Create results directory if it doesn't exist
    os.makedirs('results/closeness_centrality', exist_ok=True)
    
    # Compute closeness centrality
    closeness_centrality = nx.closeness_centrality(G)
    
    # Convert to DataFrame for easier manipulation
    closeness_df = pd.DataFrame(closeness_centrality.items(), columns=['Node', 'Closeness Centrality'])
    
    # Sort by closeness centrality in descending order
    top_3_closeness = closeness_df.sort_values(by='Closeness Centrality', ascending=False).head(3)
    
    # Print the top 3 nodes with highest closeness centrality
    print("Top 3 nodes by closeness centrality:")
    print(top_3_closeness)
    
    # Save the top 3 nodes to a CSV file
    top_3_closeness.to_csv('results/closeness_centrality/top_3_closeness_centrality.csv', index=False)

    # Plot closeness centrality distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(closeness_df['Closeness Centrality'], bins=30, kde=True)
    plt.title('Closeness Centrality Distribution')
    plt.xlabel('Closeness Centrality')
    plt.ylabel('Frequency')
    plt.savefig('results/closeness_centrality/closeness_centrality_distribution.png')


if __name__ == "__main__":
    df = pd.read_csv('twitch_gamers/large_twitch_edges.csv', nrows=10000)
    G = nx.from_pandas_edgelist(df, 'numeric_id_1', 'numeric_id_2')
    
    # Q1
    visualize_graph(G)
    # Q2
    compute_graph_metrics(G)
    # Q3
    compute_closeness_centrality(G)
