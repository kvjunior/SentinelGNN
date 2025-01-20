import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
import seaborn as sns

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

def plot_missing_data(ax):
    # Data from results
    missing_rates = np.array([0, 0.1, 0.2, 0.3, 0.4, 0.5])
    roc_auc = np.array([0.980, 0.965, 0.945, 0.920, 0.890, 0.850])
    pr_auc = np.array([0.975, 0.955, 0.930, 0.900, 0.865, 0.820])
    
    # Error margins
    roc_err = np.array([0.01, 0.015, 0.02, 0.025, 0.03, 0.035])
    pr_err = np.array([0.01, 0.015, 0.02, 0.025, 0.03, 0.035])
    
    ax.errorbar(missing_rates, roc_auc, yerr=roc_err, 
                marker='o', label='ROC-AUC', capsize=3)
    ax.errorbar(missing_rates, pr_auc, yerr=pr_err, 
                marker='s', label='PR-AUC', capsize=3)
    
    ax.set_xlabel('Missing Data Rate')
    ax.set_ylabel('Performance')
    ax.set_title('(A) Performance under Missing Data')
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend()
    ax.set_ylim(0.8, 1.0)

def plot_noise_resilience(ax):
    # Noise levels and corresponding performances
    noise_levels = np.array([0, 0.05, 0.10, 0.15, 0.20, 0.25])
    performances = {
        'SentinelGNN': np.array([0.980, 0.965, 0.945, 0.925, 0.900, 0.870]),
        'GCN': np.array([0.750, 0.720, 0.690, 0.650, 0.610, 0.580]),
        'GAT': np.array([0.720, 0.690, 0.660, 0.620, 0.580, 0.550])
    }
    
    styles = ['-o', '-s', '-^']
    for (model, perf), style in zip(performances.items(), styles):
        ax.plot(noise_levels, perf, style, label=model, markersize=5)
    
    ax.set_xlabel('Noise Level')
    ax.set_ylabel('ROC-AUC Score')
    ax.set_title('(B) Resilience to Noise')
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend()
    ax.set_ylim(0.5, 1.0)

def plot_concept_drift(ax):
    # Time periods and drift metrics
    time_periods = np.array([1, 2, 3, 4, 5, 6])
    detection_delay = np.array([1.2, 1.5, 1.3, 1.4, 1.6, 1.3])
    adaptation_rate = np.array([0.95, 0.92, 0.94, 0.93, 0.91, 0.94])
    
    ax1 = ax
    ax2 = ax1.twinx()
    
    l1 = ax1.plot(time_periods, detection_delay, 'b-o', 
                  label='Detection Delay', markersize=5)
    l2 = ax2.plot(time_periods, adaptation_rate, 'r-s', 
                  label='Adaptation Rate', markersize=5)
    
    ax1.set_xlabel('Time Period (months)')
    ax1.set_ylabel('Detection Delay (s)', color='b')
    ax2.set_ylabel('Adaptation Rate', color='r')
    
    ax1.tick_params(axis='y', labelcolor='b')
    ax2.tick_params(axis='y', labelcolor='r')
    
    # Combine legends
    lns = l1 + l2
    labels = [l.get_label() for l in lns]
    ax.legend(lns, labels, loc='lower right')
    
    ax.set_title('(C) Adaptability to Concept Drift')
    ax1.grid(True, linestyle='--', alpha=0.7)

def plot_cross_chain(ax):
    # Cross-chain performance data
    chains = ['Ethereum', 'BSC', 'Polygon', 'Arbitrum']
    metrics = {
        'Transfer': np.array([0.98, 0.96, 0.95, 0.94]),
        'Swap': np.array([0.95, 0.93, 0.92, 0.91]),
        'Lending': np.array([0.93, 0.91, 0.90, 0.89])
    }
    
    x = np.arange(len(chains))
    width = 0.25
    multiplier = 0
    
    for transaction_type, performance in metrics.items():
        offset = width * multiplier
        rects = ax.bar(x + offset, performance, width, label=transaction_type)
        multiplier += 1
    
    ax.set_ylabel('ROC-AUC Score')
    ax.set_title('(D) Cross-chain Generalization')
    ax.set_xticks(x + width)
    ax.set_xticklabels(chains, rotation=45)
    ax.legend(loc='lower left')
    ax.set_ylim(0.85, 1.0)
    ax.grid(True, linestyle='--', alpha=0.7)

def create_robustness_analysis():
    set_academic_style()
    
    # Create figure and subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
    
    # Create individual plots
    plot_missing_data(ax1)
    plot_noise_resilience(ax2)
    plot_concept_drift(ax3)
    plot_cross_chain(ax4)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure
    plt.savefig('robustness_tests.pdf', 
                bbox_inches='tight', 
                pad_inches=0.1, 
                dpi=300)
    plt.close()

if __name__ == "__main__":
    create_robustness_analysis()