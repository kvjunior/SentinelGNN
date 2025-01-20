import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.patches import Patch
import networkx as nx

def set_academic_style():
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams.update({
        'font.family': 'Times New Roman',
        'font.size': 12,           # Increased base font size
        'axes.labelsize': 14,      # Increased label size
        'axes.titlesize': 14,      # Increased title size
        'xtick.labelsize': 12,     # Increased tick label size
        'ytick.labelsize': 12,     # Increased tick label size
        'legend.fontsize': 12,     # Increased legend font size
        'figure.dpi': 300,
        'figure.figsize': (18, 6)  # Increased figure size
    })

def plot_false_positives(ax):
    categories = [
        'High-value Transfers',
        'Smart Contract Calls',
        'Flash Loans',
        'Token Swaps',
        'Batch Transactions'
    ]
    frequencies = np.array([0.023, 0.018, 0.015, 0.021, 0.019])
    confidence_scores = np.array([0.82, 0.78, 0.75, 0.73, 0.71])
    
    colors = plt.cm.RdYlBu(np.linspace(0.2, 0.8, 5))
    hatches = ['/', '\\', 'x', '.', '+']
    
    bars = ax.barh(categories, frequencies, color=colors, alpha=0.8)
    
    for bar, hatch in zip(bars, hatches):
        bar.set_hatch(hatch)
    
    # Enhanced annotations
    for i, bar in enumerate(bars):
        ax.text(bar.get_width() + 0.001, 
                bar.get_y() + bar.get_height()/2,
                f'conf: {confidence_scores[i]:.2f}',
                va='center', 
                fontsize=12,          # Increased font size
                fontweight='bold',
                bbox=dict(facecolor='white', alpha=0.8, pad=3))
    
    sm = plt.cm.ScalarMappable(cmap=plt.cm.RdYlBu, 
                              norm=plt.Normalize(vmin=0.7, vmax=0.85))
    cbar = plt.colorbar(sm, ax=ax, label='Confidence Score')
    cbar.ax.tick_params(labelsize=12)
    
    ax.set_xlabel('False Positive Rate', fontsize=14, fontweight='bold')
    ax.set_title('(A) False Positive Distribution', 
                 fontsize=14, fontweight='bold', pad=15)
    
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.tick_params(axis='both', which='major', labelsize=12)
    
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.5)

def plot_false_negatives(ax):
    n_samples = 200
    legitimate = np.random.multivariate_normal(
        [0, 0], [[1, 0.5], [0.5, 1]], n_samples)
    anomalous = np.random.multivariate_normal(
        [2, 2], [[1.5, -0.5], [-0.5, 1.5]], n_samples//4)
    missed = np.random.multivariate_normal(
        [1, 1], [[0.5, 0.2], [0.2, 0.5]], n_samples//8)
    
    # Increased marker sizes and improved colors
    ax.scatter(legitimate[:, 0], legitimate[:, 1], 
              c='#2166AC', alpha=0.7, s=50, label='Legitimate')
    ax.scatter(anomalous[:, 0], anomalous[:, 1], 
              c='#B2182B', alpha=0.7, s=50, label='Detected Anomalies')
    ax.scatter(missed[:, 0], missed[:, 1], 
              c='#F4A582', alpha=0.7, s=70, label='Missed Anomalies')
    
    theta = np.linspace(0, 2*np.pi, 100)
    r = 1.5
    ax.plot(r*np.cos(theta) + 1, r*np.sin(theta) + 1, 
            'k--', alpha=0.4, linewidth=2, label='Decision Boundary')
    
    ax.set_xlabel('Feature 1', fontsize=14)
    ax.set_ylabel('Feature 2', fontsize=14)
    ax.set_title('(B) False Negative Patterns', fontsize=14)
    ax.legend(fontsize=12, framealpha=0.9)
    ax.grid(True, alpha=0.3)

def plot_edge_cases(ax):
    G = nx.Graph()
    
    pos = {
        'Normal': (0, 0),
        'Edge1': (1, 1),
        'Edge2': (2, 0),
        'Edge3': (1, -1),
        'Anomaly': (3, 0)
    }
    
    G.add_nodes_from(pos.keys())
    G.add_edges_from([
        ('Normal', 'Edge1'),
        ('Edge1', 'Edge2'),
        ('Edge2', 'Edge3'),
        ('Edge3', 'Normal'),
        ('Edge2', 'Anomaly')
    ])
    
    confidence_scores = {
        'Normal': 0.2,
        'Edge1': 0.6,
        'Edge2': 0.7,
        'Edge3': 0.65,
        'Anomaly': 0.9
    }
    
    node_colors = [plt.cm.RdYlBu_r(score) for score in confidence_scores.values()]
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, 
                          node_size=1500, alpha=0.8)  # Increased node size
    nx.draw_networkx_edges(G, pos, edge_color='gray', 
                          alpha=0.6, width=2)  # Increased edge width
    nx.draw_networkx_labels(G, pos, font_size=12, font_weight='bold')
    
    for node, score in confidence_scores.items():
        x, y = pos[node]
        ax.text(x, y-0.25, f'{score:.2f}', 
                horizontalalignment='center', 
                fontsize=12,
                fontweight='bold',
                bbox=dict(facecolor='white', alpha=0.8, pad=2))
    
    sm = plt.cm.ScalarMappable(cmap=plt.cm.RdYlBu_r,
                              norm=plt.Normalize(vmin=0.2, vmax=0.9))
    cbar = plt.colorbar(sm, ax=ax, label='Confidence Score')
    cbar.ax.tick_params(labelsize=12)
    
    ax.set_title('(C) Edge Case Examples', fontsize=14, pad=15)
    ax.axis('off')

def create_failure_analysis():
    set_academic_style()
    
    fig = plt.figure()
    gs = fig.add_gridspec(1, 3, width_ratios=[1.2, 1, 1])
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[0, 2])
    
    plot_false_positives(ax1)
    plot_false_negatives(ax2)
    plot_edge_cases(ax3)
    
    plt.figtext(0.02, 0.98, 'Analysis Date: 2024-02-07', 
                fontsize=10, style='italic')
    
    plt.tight_layout()
    
    plt.savefig('failure_analysis.pdf', 
                bbox_inches='tight', 
                pad_inches=0.2, 
                dpi=300)
    plt.close()

if __name__ == "__main__":
    create_failure_analysis()