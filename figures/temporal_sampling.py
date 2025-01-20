import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.patches import Patch
from matplotlib.collections import LineCollection
import matplotlib.colors as mcolors

def set_academic_style():
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams.update({
        'font.family': 'Times New Roman',
        'font.size': 10,
        'axes.labelsize': 10,
        'axes.titlesize': 10,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'legend.fontsize': 9,
        'figure.dpi': 300
    })

def create_base_graph():
    G = nx.DiGraph()
    nodes = ['A', 'B', 'C', 'D']
    edges = [
        ('A', 'B', {'time': 0, 'weight': 1}),
        ('A', 'B', {'time': 1, 'weight': 2}),
        ('B', 'C', {'time': 2, 'weight': 1}),
        ('C', 'D', {'time': 3, 'weight': 3}),
        ('D', 'A', {'time': 4, 'weight': 2}),
        ('A', 'C', {'time': 5, 'weight': 1}),
    ]
    G.add_nodes_from(nodes)
    G.add_edges_from(edges)
    return G

def draw_graph_with_temporal_info(ax, G, title, pos=None, highlight_edges=None, time_windows=None):
    if pos is None:
        pos = nx.spring_layout(G, k=1, iterations=50, seed=42)
    
    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_color='lightblue', 
                          node_size=500, alpha=0.7, ax=ax)
    nx.draw_networkx_labels(G, pos, ax=ax)
    
    # Prepare edge colors based on time
    edge_times = [G[u][v]['time'] for u, v in G.edges()]
    if edge_times:
        norm = mcolors.Normalize(vmin=min(edge_times), vmax=max(edge_times))
        cmap = plt.cm.viridis
        edge_colors = [cmap(norm(t)) for t in edge_times]
    else:
        edge_colors = ['gray']
    
    # Draw edges with different styles based on context
    edges = list(G.edges())
    for i, (u, v) in enumerate(edges):
        if highlight_edges is not None and (u, v) in highlight_edges:
            ax.annotate("", xy=pos[v], xytext=pos[u],
                       arrowprops=dict(arrowstyle="->", color=edge_colors[i],
                                     linewidth=2))
        else:
            ax.annotate("", xy=pos[v], xytext=pos[u],
                       arrowprops=dict(arrowstyle="->", color=edge_colors[i],
                                     alpha=0.5))
        
        # Add edge labels with timing information
        mid_point = np.array(pos[u]) + 0.4 * (np.array(pos[v]) - np.array(pos[u]))
        ax.text(mid_point[0], mid_point[1], f"t={G[u][v]['time']}", 
                fontsize=8)
    
    # Modify time windows labeling to be clearer
    if time_windows:
        for i, window in enumerate(time_windows):
            bbox = ax.get_position()
            window_height = 0.12  # Increased height
            window_y = bbox.y0 - (i + 1) * window_height
            ax.text(bbox.x0, window_y,
                   f"Time Window {i}: [{window[0]}-{window[1]}]",  # Modified format
                   transform=ax.figure.transFigure,
                   fontsize=9,  # Increased font size
                   bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'))  # Added background box

def plot_temporal_sampling_process():
    set_academic_style()
    
    # Create figure and subplots
    fig = plt.figure(figsize=(15, 10))
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.2)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, 0])
    ax4 = fig.add_subplot(gs[1, 1])
    
    # Create base graph
    G = create_base_graph()
    pos = nx.spring_layout(G, k=1, iterations=50, seed=42)
    
    # Initial multi-edge graph
    draw_graph_with_temporal_info(ax1, G, 
                                "(A) Initial Multi-edge Graph", pos)
    
    # Time-window segmentation
    time_windows = [(0, 2), (2, 4), (4, 6)]
    draw_graph_with_temporal_info(ax2, G,
                                "(B) Time-window Segmentation", 
                                pos, time_windows=time_windows)
    
    # Importance sampling
    G_sampled = G.copy()
    important_edges = [('A', 'B'), ('C', 'D')]
    draw_graph_with_temporal_info(ax3, G_sampled,
                                "(C) Importance Sampling",
                                pos, highlight_edges=important_edges)
    
    # Final sampled edges
    G_final = nx.DiGraph()
    G_final.add_nodes_from(G.nodes())
    final_edges = [
        ('A', 'B', {'time': 1, 'weight': 2}),
        ('B', 'C', {'time': 2, 'weight': 1}),
        ('C', 'D', {'time': 3, 'weight': 3}),
    ]
    G_final.add_edges_from(final_edges)
    draw_graph_with_temporal_info(ax4, G_final,
                                "(D) Final Sampled Edges", 
                                pos)
    
    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis, 
                              norm=plt.Normalize(vmin=0, vmax=5))
    cbar = plt.colorbar(sm, ax=[ax1, ax2, ax3, ax4], 
                       label='Timestamp', orientation='horizontal',
                       fraction=0.02, pad=0.04)
    
    # Add title
    plt.suptitle('Temporal-aware Edge Sampling Process', 
                y=1.02, fontsize=12)
    
    # Save figure
    plt.savefig('temporal_sampling.pdf',
                bbox_inches='tight',
                pad_inches=0.1,
                dpi=300)
    plt.close()

if __name__ == "__main__":
    plot_temporal_sampling_process()