'''
Author: danielwangow daomiao.wang@live.com
Date: 2025-06-17 21:14:33
LastEditors: danielwangow daomiao.wang@live.com
LastEditTime: 2025-12-12 22:02:20
FilePath: /TDA-Homology/utils/dataPloter.py
Description: 
-----> VENI VIDI VICI <-----
Copyright (c) 2025 by Daniel.Wang@Fudan University. All Rights Reserved.
'''

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
import os

# Plot comparison between normal and noisy PPG signals with noise position indicator.
def plot_data_profile(normal_signal, noisy_signal, label_positions, save_path=None):
    # Create figure with 3 subplots vertically
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(6, 4))

    # Set style for academic presentation
    plt.style.use('seaborn-v0_8-paper')
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.serif'] = ['Times New Roman']

    # Plot normal PPG signal
    ax1.plot(normal_signal, color='#2E86C1', linewidth=1.5)
    ax1.set_title('Normal PPG Waveform', fontsize=12, fontweight='bold', fontfamily='Times New Roman')
    ax1.set_ylabel('Amplitude', fontsize=10, fontfamily='Times New Roman')
    ax1.set_ylim(-1.5, 2.5)
    ax1.set_yticks([-1, 0, 1, 2])
    ax1.grid(linestyle='--', alpha=0.7)

    # Plot noisy PPG signal  
    ax2.plot(noisy_signal, color='#E74C3C', linewidth=1.5)
    ax2.set_title('Anomaly PPG Waveform (Skew)', fontsize=12, fontweight='bold', fontfamily='Times New Roman')
    ax2.set_ylabel('Amplitude', fontsize=10, fontfamily='Times New Roman')
    ax2.set_ylim(-1.5, 2.5)
    ax2.set_yticks([-1, 0, 1, 2])
    ax2.grid(linestyle='--', alpha=0.7)
    
    # Plot noise position indicator
    ax3.plot(label_positions, color='#000000', linewidth=1.5)
    ax3.set_title('Position Indicator of Anomalies', fontsize=12, fontweight='bold', fontfamily='Times New Roman')
    ax3.set_ylabel('Indicator', fontsize=10, fontfamily='Times New Roman')
    ax3.set_ylim(-0.2, 1.2)
    ax3.set_yticks([0, 1])
    ax3.grid(linestyle='--', alpha=0.7)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=600, bbox_inches='tight', transparent=True)
    plt.show()

def plot_data_checker(sig1, sig2, sig3, sig4):
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(6, 6))
    ax1.plot(sig1, color='#2E86C1', linewidth=1.5)
    ax1.set_ylabel('Amplitude', fontsize=10, fontfamily='Times New Roman')
    ax1.grid(linestyle='--', alpha=0.7)

    ax2.plot(sig2, color='#E74C3C', linewidth=1.5)
    ax2.set_ylabel('Amplitude', fontsize=10, fontfamily='Times New Roman')
    ax2.grid(linestyle='--', alpha=0.7)

    ax3.plot(sig3, color='#2E86C1', linewidth=1.5)
    ax3.set_ylabel('Amplitude', fontsize=10, fontfamily='Times New Roman')
    ax3.grid(linestyle='--', alpha=0.7)

    ax4.plot(sig4, color='#E74C3C', linewidth=1.5)
    ax4.set_ylabel('Amplitude', fontsize=10, fontfamily='Times New Roman')
    ax4.grid(linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

def visualize_delay_embedding_simple(window, cloud, subwindow_dim):
    """Optimized visualization function with academic-style formatting"""
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    from matplotlib.font_manager import FontProperties
    from matplotlib.collections import LineCollection
    plt.style.use('seaborn-v0_8-paper')
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['mathtext.fontset'] = 'stix'
    font = FontProperties(family=['Times New Roman'])

    fig = plt.figure(figsize=(9,4), dpi=600)
    gs = fig.add_gridspec(1, 2)

    ax1 = fig.add_subplot(gs[0, 0])
    points = np.column_stack((np.arange(len(window)), window))
    segments = np.column_stack((points[:-1], points[1:]))
    norm = plt.Normalize(0, len(window)-1)
    lc = LineCollection(segments.reshape(-1, 2, 2), cmap='jet', norm=norm)
    lc.set_array(np.arange(len(window)-1))
    lc.set_linewidth(2)
    ax1.add_collection(lc)
    ax1.set_title('Time Series', fontsize=10, fontweight='bold',fontfamily='Times New Roman')
    ax1.set_ylabel('Amplitude', fontsize=8, fontfamily='Times New Roman')
    #ax1.set_ylim(-1.5, 2.5)
    #ax1.set_yticks([-1, 0, 1, 2])
    ax1.grid(linestyle='--', alpha=0.7)
    ax1.autoscale_view()

    if subwindow_dim == 2:
        ax3 = fig.add_subplot(gs[0, 1])
        scatter = ax3.scatter(cloud[:, 0], cloud[:, 1], 
                            c=np.arange(len(cloud)), cmap='jet',
                            alpha=0.8, s=30, 
                            edgecolors='white', linewidth=0.5)
        ax3.set_title('2D Delay Embedding', fontsize=10, fontweight='bold',fontfamily='Times New Roman')
        ax3.set_xlabel('$x(t)$', fontsize=8, fontfamily='Times New Roman')
        ax3.set_ylabel('$x(t+\\tau)$', fontsize=8, fontfamily='Times New Roman')
        ax3.grid(linestyle='--', alpha=0.7)
        ax3.autoscale_view()
        
        # Draw lines connecting consecutive points
        for i in range(len(cloud)-1):
            ax3.plot(cloud[i:i+2, 0], cloud[i:i+2, 1], 
                    color=plt.cm.jet(i/(len(cloud)-1)), 
                    alpha=0.3, linewidth=1)
        plt.colorbar(scatter, ax=[ax1, ax3], label='Time Step', orientation='horizontal', shrink=0.7, pad=0.08)

    elif subwindow_dim == 3:
        ax3 = fig.add_subplot(gs[0, 1], projection='3d')
        ax3.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.1))
        ax3.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.1))
        ax3.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.1))
        scatter = ax3.scatter(cloud[:, 0], cloud[:, 1], cloud[:, 2],
                            c=np.arange(len(cloud)), cmap='jet',
                            alpha=0.8, s=50,
                            edgecolors='white', linewidth=0.5)
        ax3.set_title('3D Delay Embedding', fontsize=10, fontweight='bold', fontfamily='Times New Roman')
        ax3.set_xlabel('$x(t)$', fontsize=8, fontfamily='Times New Roman')
        ax3.set_ylabel('$x(t+\\tau)$', fontsize=8, fontfamily='Times New Roman')
        ax3.set_zlabel('$x(t+2\\tau)$', fontsize=8, fontfamily='Times New Roman')
        ax3.grid(True, linestyle='--', alpha=0.7)
        ax3.view_init(elev=30, azim=45)
        plt.colorbar(scatter, ax=[ax1, ax3], label='Time Step', orientation='horizontal', shrink=0.7, pad=0.08)

    else:
        ax3 = fig.add_subplot(gs[0, 1], projection='3d')
        ax3.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.1))
        ax3.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.1))
        ax3.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.1))
        if cloud.shape[0] > 0:
            pca = PCA(n_components=min(3, cloud.shape[1]))
            pca_cloud = pca.fit_transform(cloud)
            if pca_cloud.shape[1] >= 3:
                scatter2 = ax3.scatter(pca_cloud[:, 0], pca_cloud[:, 1], pca_cloud[:, 2],
                                   c=np.arange(len(pca_cloud)), cmap='jet',
                                   alpha=0.6, s=30,
                                   edgecolors='white', linewidth=0.5)
            else:
                scatter2 = ax3.scatter(pca_cloud[:, 0], pca_cloud[:, 1],
                                   np.zeros(len(pca_cloud)),
                                   c=np.arange(len(pca_cloud)), cmap='jet',
                                   alpha=0.6, s=30,
                                   edgecolors='white', linewidth=0.5)
            ax3.set_title(f'{subwindow_dim}D Embedding\n(PCA Visualization)', fontsize=10, fontweight='bold', fontfamily='Times New Roman')
            ax3.set_xlabel(f'PC1 {pca.explained_variance_ratio_[0]:.1%}', fontsize=8, fontfamily='Times New Roman')
            ax3.set_ylabel(f'PC2 {pca.explained_variance_ratio_[1]:.1%}', fontsize=8, fontfamily='Times New Roman')
            if pca_cloud.shape[1] >= 3:
                ax3.set_zlabel(f'PC3 {pca.explained_variance_ratio_[2]:.1%}', fontsize=8, fontfamily='Times New Roman')
            ax3.grid(linestyle='--', alpha=0.8)
            ax3.tick_params(labelsize=8)
            ax3.view_init(elev=30, azim=45)
            plt.colorbar(scatter2, ax=ax1, label='Time Step', orientation='horizontal', pad=0.1)
    fig.patch.set_facecolor('white')
    plt.show()

def visualize_anomaly_scores(timeseries, anomaly_scores):
    try:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(4, 4),  dpi=600)
        plt.style.use('seaborn-v0_8-paper')
        plt.rcParams['font.family'] = 'Times New Roman'
        plt.rcParams['font.serif'] = ['Times New Roman']
        #plt.rcParams['mathtext.fontset'] = 'stix'
    
        ax1.plot(timeseries, color='#2E86C1', linewidth=1.5)
        ax1.set_title('Input Signal', fontsize=10, fontweight='bold', fontfamily='Times New Roman')
        ax1.set_ylabel('Amplitude', fontsize=8, fontfamily='Times New Roman')
        ax1.set_ylim(-1.5, 2.5)
        ax1.set_yticks([-1, 0, 1, 2])
        ax1.set_xticks([0, len(timeseries)*0.25, len(timeseries)*0.5, len(timeseries)*0.75, len(timeseries)-1])
        ax1.set_xticklabels(['0', '25%', '50%', '75%', '100%'])
        ax1.grid(linestyle='--', alpha=0.7)
        
        ax2.plot(anomaly_scores, 'g-', linewidth=1.5, label='Anomaly Scores')
        
        # Calculate threshold and get anomaly regions
        threshold = np.percentile(anomaly_scores, 90)
        high_anomaly_mask = anomaly_scores > threshold
        
        # Get continuous anomaly regions
        anomaly_regions = []
        start_idx = None
        for i in range(len(high_anomaly_mask)):
            if high_anomaly_mask[i] and start_idx is None:
                start_idx = i
            elif not high_anomaly_mask[i] and start_idx is not None:
                anomaly_regions.append((start_idx, i))
                start_idx = None
        if start_idx is not None:
            anomaly_regions.append((start_idx, len(high_anomaly_mask)))

        # Fill anomaly regions
        ax2.fill_between(range(len(anomaly_scores)), 0, anomaly_scores,
                        where=high_anomaly_mask, alpha=0.5, color='red',
                        label=f'Anomaly Region > {threshold:.3f}',
                        linewidth=0)
        
        ax2.set_title('Predicted Anomaly Scores', fontsize=10, fontweight='bold', fontfamily='Times New Roman')
        ax2.set_ylabel('Scores', fontsize=8, fontfamily='Times New Roman')
        ax2.set_xticks([0, len(timeseries)*0.25, len(timeseries)*0.5, len(timeseries)*0.75, len(timeseries)-1])
        ax2.set_xticklabels(['0', '25%', '50%', '75%', '100%'])
        ax2.legend(fontsize=6)
        ax2.grid(linestyle='--', alpha=0.7)
        
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.show()
        return anomaly_regions
    
    except Exception as e:
        print(f"Visualization failed: {e}")

def visualize_delay_embedding_full(window, cloud, indicator, subwindow_dim):
    """Optimized visualization function with academic-style formatting"""
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    from matplotlib.font_manager import FontProperties
    from matplotlib.collections import LineCollection
    plt.style.use('seaborn-v0_8-paper')
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['mathtext.fontset'] = 'stix'
    font = FontProperties(family=['Times New Roman'])

    fig = plt.figure(figsize=(9,3), dpi=600)
    gs = fig.add_gridspec(1, 3)

    ax1 = fig.add_subplot(gs[0, 0])
    points = np.column_stack((np.arange(len(window)), window))
    segments = np.column_stack((points[:-1], points[1:]))
    norm = plt.Normalize(0, len(window)-1)
    lc = LineCollection(segments.reshape(-1, 2, 2), cmap='jet', norm=norm)
    lc.set_array(np.arange(len(window)-1))
    lc.set_linewidth(2)
    ax1.add_collection(lc)
    ax1.set_title('Time Series', fontsize=10, fontweight='bold',fontfamily='Times New Roman')
    ax1.set_ylabel('Amplitude', fontsize=8, fontfamily='Times New Roman')
    ax1.set_ylim(-1.5, 2.5)
    ax1.set_yticks([-1, 0, 1, 2])
    ax1.grid(linestyle='--', alpha=0.7)
    ax1.autoscale_view()

    ax2 = fig.add_subplot(gs[0, 1])
    labels = np.column_stack((np.arange(len(window)), indicator))
    segments = np.column_stack((labels[:-1], labels[1:]))
    norm = plt.Normalize(0, len(window)-1)
    lc = LineCollection(segments.reshape(-1, 2, 2), cmap='jet', norm=norm)
    lc.set_array(np.arange(len(window)-1))
    lc.set_linewidth(2)
    ax2.add_collection(lc)
    ax2.set_title('Position Indicator of Anomalies', fontsize=10, fontweight='bold',fontfamily='Times New Roman')
    ax2.set_ylabel('Indicator', fontsize=8, fontfamily='Times New Roman')
    ax2.set_ylim(-0.1, 1.1)
    ax2.set_yticks([0, 1])
    ax2.grid(linestyle='--', alpha=0.7)
    ax2.autoscale_view()

    if subwindow_dim == 2:
        ax3 = fig.add_subplot(gs[0, 2])
        scatter = ax3.scatter(cloud[:, 0], cloud[:, 1], 
                            c=np.arange(len(cloud)), cmap='jet',
                            alpha=0.8, s=30, 
                            edgecolors='white', linewidth=0.5)
        ax3.set_title('2D Delay Embedding', fontsize=10, fontweight='bold',fontfamily='Times New Roman')
        ax3.set_xlabel('$x(t)$', fontsize=8, fontfamily='Times New Roman')
        ax3.set_ylabel('$x(t+\\tau)$', fontsize=8, fontfamily='Times New Roman')
        ax3.grid(linestyle='--', alpha=0.7)
        ax3.autoscale_view()
        
        # Draw lines connecting consecutive points
        for i in range(len(cloud)-1):
            ax3.plot(cloud[i:i+2, 0], cloud[i:i+2, 1], 
                    color=plt.cm.jet(i/(len(cloud)-1)), 
                    alpha=0.3, linewidth=1)
        plt.colorbar(scatter, ax=[ax1,ax2, ax3], label='Time Step', orientation='horizontal', shrink=0.7, pad=0.08)

    elif subwindow_dim == 3:
        ax3 = fig.add_subplot(gs[0, 2], projection='3d')
        ax3.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.1))
        ax3.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.1))
        ax3.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.1))
        scatter = ax3.scatter(cloud[:, 0], cloud[:, 1], cloud[:, 2],
                            c=np.arange(len(cloud)), cmap='jet',
                            alpha=0.8, s=50,
                            edgecolors='white', linewidth=0.5)
        ax3.set_title('3D Delay Embedding', fontsize=10, fontweight='bold', fontfamily='Times New Roman')
        ax3.set_xlabel('$x(t)$', fontsize=8, fontfamily='Times New Roman')
        ax3.set_ylabel('$x(t+\\tau)$', fontsize=8, fontfamily='Times New Roman')
        ax3.set_zlabel('$x(t+2\\tau)$', fontsize=8, fontfamily='Times New Roman')
        ax3.grid(True, linestyle='--', alpha=0.7)
        ax3.view_init(elev=30, azim=45)
        plt.colorbar(scatter, ax=[ax1,ax2, ax3], label='Time Step', orientation='horizontal', shrink=0.7, pad=0.08)

    else:
        ax3 = fig.add_subplot(gs[0, 2], projection='3d')
        ax3.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.1))
        ax3.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.1))
        ax3.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.1))
        if cloud.shape[0] > 0:
            pca = PCA(n_components=min(3, cloud.shape[1]))
            pca_cloud = pca.fit_transform(cloud)
            if pca_cloud.shape[1] >= 3:
                scatter2 = ax3.scatter(pca_cloud[:, 0], pca_cloud[:, 1], pca_cloud[:, 2],
                                   c=np.arange(len(pca_cloud)), cmap='jet',
                                   alpha=0.6, s=30,
                                   edgecolors='white', linewidth=0.5)
            else:
                scatter2 = ax3.scatter(pca_cloud[:, 0], pca_cloud[:, 1],
                                   np.zeros(len(pca_cloud)),
                                   c=np.arange(len(pca_cloud)), cmap='jet',
                                   alpha=0.6, s=30,
                                   edgecolors='white', linewidth=0.5)
            ax3.set_title(f'{subwindow_dim}D Embedding\n(PCA Visualization)', fontsize=10, fontweight='bold', fontfamily='Times New Roman')
            ax3.set_xlabel(f'PC1 {pca.explained_variance_ratio_[0]:.1%}', fontsize=8, fontfamily='Times New Roman')
            ax3.set_ylabel(f'PC2 {pca.explained_variance_ratio_[1]:.1%}', fontsize=8, fontfamily='Times New Roman')
            if pca_cloud.shape[1] >= 3:
                ax3.set_zlabel(f'PC3 {pca.explained_variance_ratio_[2]:.1%}', fontsize=8, fontfamily='Times New Roman')
            ax3.grid(linestyle='--', alpha=0.8)
            ax3.tick_params(labelsize=8)
            ax3.view_init(elev=30, azim=45)
            plt.colorbar(scatter2, ax=[ax1, ax2, ax3], label='Time Step', orientation='horizontal', shrink=0.55, pad=0.25)
    fig.patch.set_facecolor('white')
    plt.show()

def plot_cycles_with_anomalies(L, cycles, anomaly_cycles, d, pc=None, save_path=None):
    try:
        plt.style.use('seaborn-v0_8-paper')
        plt.rcParams['font.family'] = 'Times New Roman'
        plt.rcParams['mathtext.fontset'] = 'stix'
        
        if d == 2:
            fig = plt.figure(figsize=(6, 6), dpi=600)
            fig.suptitle('Normal and Abnormal Cycles', 
                       fontsize=12, fontweight='bold', fontfamily='Times New Roman')
            for cycle in cycles:
                n = len(cycle)
                for i in range(n):
                    xs_v, ys_v = L[[cycle[i][0], cycle[i][1]]][:, 0], L[[cycle[i][0], cycle[i][1]]][:, 1]
                    plt.plot(xs_v, ys_v, color='blue', linewidth=2, alpha=0.8)
            c = 1
            for cycle in anomaly_cycles:
                n = len(cycle)
                for i in range(n):
                    xs_v, ys_v = L[[cycle[i][0], cycle[i][1]]][:, 0], L[[cycle[i][0], cycle[i][1]]][:, 1]
                    plt.plot(xs_v, ys_v, color='red', linewidth=3, alpha=0.9)
                c += 1
            
            if pc is not None:
                plt.scatter(pc[:, 0], pc[:, 1], color='black', s=20, alpha=0.6, label='Point Cloud')
            else:
                plt.scatter(L[:, 0], L[:, 1], color='black', s=20, alpha=0.6, label='Subsampled Points')
            
            plt.xlabel('$x(t)$', fontsize=10, fontfamily='Times New Roman')
            plt.ylabel('$x(t+\\tau)$', fontsize=10, fontfamily='Times New Roman')
            plt.grid(linestyle='--', alpha=0.7)
            plt.legend()
            
        elif d == 3:
            fig = plt.figure(figsize=(8, 6), dpi=600)
            fig.suptitle('Normal and Abnormal Cycles', 
                       fontsize=12, fontweight='bold', fontfamily='Times New Roman')
            ax = fig.add_subplot(111, projection='3d')
            for cycle in cycles:
                n = len(cycle)
                for i in range(n):
                    xs_v, ys_v, zs_v = L[[cycle[i][0], cycle[i][1]]][:, 0], L[[cycle[i][0], cycle[i][1]]][:, 1], L[[cycle[i][0], cycle[i][1]]][:, 2]
                    ax.plot(xs_v, ys_v, zs_v, color='blue', linewidth=2, alpha=0.8)
            c = 1
            for cycle in anomaly_cycles:
                n = len(cycle)
                for i in range(n):
                    xs_v, ys_v, zs_v = L[[cycle[i][0], cycle[i][1]]][:, 0], L[[cycle[i][0], cycle[i][1]]][:, 1], L[[cycle[i][0], cycle[i][1]]][:, 2]
                    ax.plot(xs_v, ys_v, zs_v, color='red', linewidth=3, alpha=0.9)
                c += 1
            if pc is not None:
                ax.scatter(pc[:, 0], pc[:, 1], pc[:, 2], color='black', s=20, alpha=0.6, label='Point Cloud')
            else:
                ax.scatter(L[:, 0], L[:, 1], L[:, 2], color='black', s=20, alpha=0.6, label='Subsampled Points')
            ax.set_xlabel('$x(t)$', fontsize=10, fontfamily='Times New Roman')
            ax.set_ylabel('$x(t+\\tau)$', fontsize=10, fontfamily='Times New Roman')
            ax.set_zlabel('$x(t+2\\tau)$', fontsize=10, fontfamily='Times New Roman')
            ax.grid(True, linestyle='--', alpha=0.3)
            ax.legend()
            ax.view_init(elev=30, azim=45)
            
        else:  
            fig = plt.figure(figsize=(6, 6), dpi=600, tight_layout=True)
            fig.suptitle('Normal and Abnormal Cycles', fontsize=12, fontweight='bold', fontfamily='Times New Roman')
            ax = fig.add_subplot(111, projection='3d')
            ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.1))
            ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.1))
            ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.1))

            pca = PCA(n_components=3)
            pca_L = pca.fit_transform(L)
            if pc is not None:
                pca_pc = pca.transform(pc)
            
            for cycle_idx, cycle in enumerate(cycles):
                n = len(cycle)
                for i in range(n):
                    xs_v, ys_v, zs_v = pca_L[[cycle[i][0], cycle[i][1]]][:, 0], pca_L[[cycle[i][0], cycle[i][1]]][:, 1], pca_L[[cycle[i][0], cycle[i][1]]][:, 2]
                    ax.plot(xs_v, ys_v, zs_v, color='blue', linewidth=3.5, alpha=0.8, 
                           label='Normal Cycle' if cycle_idx == 0 and i == 0 else None)
            
            c = 1
            for cycle_idx, cycle in enumerate(anomaly_cycles):
                n = len(cycle)
                for i in range(n):
                    xs_v, ys_v, zs_v = pca_L[[cycle[i][0], cycle[i][1]]][:, 0], pca_L[[cycle[i][0], cycle[i][1]]][:, 1], pca_L[[cycle[i][0], cycle[i][1]]][:, 2]
                    ax.plot(xs_v, ys_v, zs_v, color='red', linewidth=4.5, alpha=0.9, 
                           label='Abnormal Cycle' if cycle_idx == 0 and i == 0 else None)
                c += 1
            
            if pc is not None:
                ax.scatter(pca_L[:, 0], pca_L[:, 1], pca_L[:, 2], color='0.3', s=20, alpha=0.5, label='Point Cloud')
            else:
                ax.scatter(pca_L[:, 0], pca_L[:, 1], pca_L[:, 2], color='0.3', s=20, alpha=0.5, label='Subsampled Points')
            
            ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})', fontsize=8, fontfamily='Times New Roman')
            ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})', fontsize=8, fontfamily='Times New Roman')
            ax.set_zlabel(f'PC3 ({pca.explained_variance_ratio_[2]:.1%})', fontsize=8, fontfamily='Times New Roman')
            ax.grid(linestyle='--', alpha=0.7)
            ax.tick_params(labelsize=8)
            ax.legend(fontsize=8)
            ax.view_init(elev=30, azim=45)
        
        plt.savefig("img4concept/cycles_with_anomalies.pdf", dpi=600, bbox_inches='tight', transparent=True)
        if save_path:
            plt.savefig(save_path, dpi=600, bbox_inches='tight', transparent=True)
            print(f"Cycle visualization saved to: {save_path}")
        plt.show()
        
    except Exception as e:
        print(f"Cycle visualization failed: {e}")
        import traceback
        traceback.print_exc()

def plot_persistence_diagram_local(P, diagram):
    try:
        plt.style.use('seaborn-v0_8-paper')
        plt.rcParams['font.family'] = 'Times New Roman'
        plt.rcParams['mathtext.fontset'] = 'stix'
        
        fig = plt.figure(figsize=(8, 4), dpi=600)
        ax1 = fig.add_subplot(1, 2, 1, projection='3d')
        ax2 = fig.add_subplot(1, 2, 2)

        ax1.plot([0, 1], [0, 1], 'k--', alpha=0.5, linewidth=1)
        ax1.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.1))
        ax1.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.1))
        ax1.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.1))
        if P.shape[1] >= 3:
            scatter1 = ax1.scatter(P[:, 0], P[:, 1], P[:, 2],
                                c=np.arange(len(P)), cmap='jet',
                                alpha=0.8, s=50,
                                edgecolors='white', linewidth=0.5)
        else:
            pca = PCA(n_components=3)
            P_pca = pca.fit_transform(diagram)
            scatter1 = ax1.scatter(P_pca[:, 0], P_pca[:, 1], P_pca[:, 2],
                                c=np.arange(len(P_pca)), cmap='jet',
                                alpha=0.8, s=70,
                                edgecolors='white', linewidth=0.5)
        ax1.set_title('Subsampled Point Cloud (PCA)', fontsize=10, fontweight='bold', fontfamily='Times New Roman')
        ax1.set_xlabel('PC1', fontsize=8, fontfamily='Times New Roman')
        ax1.set_ylabel('PC2', fontsize=8, fontfamily='Times New Roman')
        ax1.set_zlabel('PC3', fontsize=8, fontfamily='Times New Roman')
        ax1.grid(True, linestyle='--', alpha=0.9)
        ax1.tick_params(labelsize=8)
        ax1.view_init(elev=30, azim=45)
        plt.colorbar(scatter1, ax=ax1, label='Time Step', orientation='horizontal', shrink=0.5, pad=0.08)
        ax1.set_aspect('equal')
        
        if len(diagram) > 0:
            births = diagram[:, 0]
            deaths = diagram[:, 1]
            min_val = min(np.min(births), np.min(deaths))
            max_val = max(np.max(births), np.max(deaths))
            
            # Round min_val down to nearest .5
            min_val = np.floor(min_val * 2) / 2
            max_val = np.ceil(max_val * 2) / 2
            # Fill the area below diagonal
            diagonal_x = np.array([min_val, max_val])
            diagonal_y = np.array([min_val, max_val])
            ax2.fill_between(diagonal_x, diagonal_y, min_val, color='gray', alpha=0.3)
            ax2.fill_between(diagonal_x, min_val, diagonal_y, color='gray', alpha=0.3)
            
            # Plot diagonal line
            ax2.plot([min_val, max_val], [min_val, max_val], 'k-', alpha=0.7, linewidth=1.5)
            scatter = ax2.scatter(births, deaths, c=deaths - births, cmap='viridis', 
                               s=70, alpha=0.7, edgecolors='black', linewidth=0.5)
            # cbar = plt.colorbar(scatter, ax=ax2)
            # cbar.set_label('Persistence', fontsize=10, fontweight='bold',fontfamily='Times New Roman')
            ax2.set_xlabel('Birth Time', fontsize=12, fontfamily='Times New Roman')
            ax2.set_ylabel('Death Time', fontsize=12, fontfamily='Times New Roman')
            ax2.set_title('Persistence Diagram', fontsize=12, fontweight='bold', fontfamily='Times New Roman')
            #ax2.grid(linestyle='--', alpha=0.7)
            ax2.grid(False)
            ax2.set_aspect('equal')
        else:
            ax2.text(0.5, 0.5, 'No persistent features detected', 
                   ha='center', va='center', transform=ax2.transAxes,
                   fontsize=12, fontfamily='Times New Roman')
            ax2.set_title('Persistence Diagram', fontsize=12, fontweight='bold', fontfamily='Times New Roman')
        
        plt.savefig("img4concept/persistence_diagram_local.pdf", dpi=600, bbox_inches='tight', transparent=True)
        
        plt.tight_layout()
        plt.show()
        
    except Exception as e:
        print(f"Persistence diagram visualization failed: {e}")
        import traceback
        traceback.print_exc()

def tda_analysis_results(timeseries, anomaly_scores, percentile, true_anomalies=None, save_path=None, show=True):
    if show:
        _tda_analysis_results(timeseries, anomaly_scores, percentile, true_anomalies, save_path)
    else:
        return _tda_analysis_results(timeseries, anomaly_scores, percentile, true_anomalies, save_path)

def _tda_analysis_results(timeseries, anomaly_scores, percentile=90,true_anomalies=None, save_path=None):
    try:
        if timeseries is None or anomaly_scores is None:
            print("Error: timeseries or anomaly_scores is None")
            return
        timeseries = np.asarray(timeseries)
        anomaly_scores = np.asarray(anomaly_scores)
        if len(timeseries) == 0 or len(anomaly_scores) == 0:
            print("Error: input array is empty")
            return
        min_length = min(len(timeseries), len(anomaly_scores))
        timeseries = timeseries[:min_length]
        anomaly_scores = anomaly_scores[:min_length]
        if np.any(np.isnan(timeseries)) or np.any(np.isinf(timeseries)):
            print("Warning: timeseries contains NaN or inf values, will be replaced with 0")
            timeseries = np.nan_to_num(timeseries, nan=0.0, posinf=0.0, neginf=0.0)
        if np.any(np.isnan(anomaly_scores)) or np.any(np.isinf(anomaly_scores)):
            print("Warning: anomaly_scores contains NaN or inf values, will be replaced with 0")
            anomaly_scores = np.nan_to_num(anomaly_scores, nan=0.0, posinf=0.0, neginf=0.0)
        plt.style.use('seaborn-v0_8-paper')
        plt.rcParams['font.family'] = 'Times New Roman'
        plt.rcParams['font.serif'] = ['Times New Roman']
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(6, 5), dpi=600)
        x_values = np.arange(len(timeseries))
        ax1.plot(x_values, timeseries, color='#2E86C1', linewidth=2) 
        ax1.set_title('Input Signal', fontsize=10, fontweight='bold', fontfamily='Times New Roman')
        ax1.set_ylabel('Amplitude', fontsize=8, fontfamily='Times New Roman')
        ax1.grid(linestyle='--', alpha=0.7)
        ax1.set_xticks([0, len(timeseries)*0.25, len(timeseries)*0.5, len(timeseries)*0.75, len(timeseries)-1])
        ax1.set_xticklabels(['0', '25%', '50%', '75%', '100%'])
        ax1.set_ylim(timeseries.min() - 0.1 * (timeseries.max() - timeseries.min()),
                    timeseries.max() + 0.1 * (timeseries.max() - timeseries.min()))
        x_values = np.arange(len(anomaly_scores))
        ax2.plot(x_values, anomaly_scores, color='red', linewidth=2.5, label='Score')
        if len(anomaly_scores) > 0:
            threshold = np.percentile(anomaly_scores, percentile)
            high_anomaly_mask = anomaly_scores > threshold
            ax2.fill_between(x_values, 0, anomaly_scores, 
                            where=high_anomaly_mask, alpha=0.3, color='red', label='Region')
            ax2.axhline(y=threshold, color='black', linestyle=':', linewidth=1.5,  
                       label=f'Threshold ({threshold:.3f})')
        ax2.set_title('Detected Anomaly Scores', fontsize=10, fontweight='bold', fontfamily='Times New Roman')
        ax2.set_ylabel('Score', fontsize=8, fontfamily='Times New Roman')
        ax2.set_xticks([0, len(anomaly_scores)*0.25, len(anomaly_scores)*0.5, len(anomaly_scores)*0.75, len(anomaly_scores)-1])
        ax2.set_xticklabels(['0', '25%', '50%', '75%', '100%'])
        #ax2.grid(linestyle='--', alpha=0.7)
        ax2.grid(False)
        # ax2.legend(fontsize=8, ncol=3)
        if len(anomaly_scores) > 0:
            threshold = np.percentile(anomaly_scores, percentile)
            anomaly_indicator = (anomaly_scores > threshold).astype(float)
        else:
            anomaly_indicator = np.zeros(len(anomaly_scores))
            
        ax3.plot(x_values, anomaly_indicator, color='#000000', 
                 linewidth=1.5, alpha=0.8, label='Detected Anomalies')
        if true_anomalies is not None and len(true_anomalies) > 0:
            for start, end in true_anomalies:
                mask = np.zeros(len(timeseries), dtype=bool)
                mask[start:end+1] = True
                ax3.fill_between(x_values, np.min(timeseries), np.max(timeseries), 
                                 where=mask, alpha=0.3, color='#E74C3C', label='True Anomalies')
        ax3.set_title('Indicator of Anomalies', fontsize=10, fontweight='bold', fontfamily='Times New Roman')
        ax3.set_ylabel('Indicator', fontsize=8, fontfamily='Times New Roman')
        ax3.set_ylim(-0.1, 1.1)
        ax3.set_yticks([0, 1])
        ax3.set_xticks([0, len(anomaly_indicator)*0.25, len(anomaly_indicator)*0.5, len(anomaly_indicator)*0.75, len(anomaly_indicator)-1])
        ax3.set_xticklabels(['0', '25%', '50%', '75%', '100%'])
        ax3.grid(linestyle='--', alpha=0.7)
        ax3.legend(fontsize=8)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(os.path.join(save_path, 'img2TDA_analysis.pdf'), dpi=600, bbox_inches='tight', transparent=True)
            print(f"TDA analysis results saved to: {save_path}")
        plt.show()
        print(f"Visualization successful: time series length={len(timeseries)}")
        if len(anomaly_scores) > 0:
            print(f"Anomaly score statistics: mean={np.mean(anomaly_scores):.4f}, std={np.std(anomaly_scores):.4f}")
            print(f"Number of detected anomalies: {np.sum(anomaly_indicator)}")
    except Exception as e:
        print(f"TDA analysis visualization failed: {e}")
        import traceback
        traceback.print_exc()