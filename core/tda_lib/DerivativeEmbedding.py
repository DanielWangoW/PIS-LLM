'''
Author: danielwangow daomiao.wang@live.com
Date: 2025-01-17 10:00:00
LastEditors: danielwangow daomiao.wang@live.com
LastEditTime: 2025-01-17 10:00:00
FilePath: /TDA-Homology/DerivativeEmbedding.py
Description: Derivative embedding for time series analysis
-----> VENI VIDI VICI <-----
Copyright (c) 2025 by Daniel.Wang@Fudan University. All Rights Reserved.
'''

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from sklearn.decomposition import PCA
import multiprocessing as mp
import warnings
warnings.filterwarnings('ignore')

class DerivativeEmbeddingProcessor:
    """
    基于导数嵌入的时间序列处理器
    用于心血管信号(ECG/PPG)的拓扑数据分析
    """
    
    def __init__(self, n_jobs=-1, derivative_order=1, smoothing_window=5):
        self.n_jobs = n_jobs if n_jobs > 0 else mp.cpu_count()
        self.derivative_order = derivative_order
        self.smoothing_window = smoothing_window
        
    def compute_derivatives(self, signal_data, order=1):
        """计算时间序列的数值导数"""
        if order == 1:
            derivative = np.gradient(signal_data)
        elif order == 2:
            derivative = np.gradient(np.gradient(signal_data))
        else:
            derivative = signal_data
            for _ in range(order):
                derivative = np.gradient(derivative)
        return derivative
    
    def smooth_signal(self, signal_data, window_size=None):
        """平滑信号以减少噪声"""
        if window_size is None:
            window_size = self.smoothing_window
        if window_size <= 1:
            return signal_data
        try:
            smoothed = signal.savgol_filter(signal_data, window_size, 2)
        except:
            kernel = np.ones(window_size) / window_size
            smoothed = np.convolve(signal_data, kernel, mode='same')
        return smoothed
    
    def compute_derivative_embedding(self, window, embedding_dim=2, derivative_orders=None, 
                                   smoothing=True, show=False):
        """计算导数嵌入点云"""
        window = np.asarray(window)
        n = len(window)
        
        if derivative_orders is None:
            derivative_orders = list(range(min(embedding_dim, 3)))
        
        embedding_dim = min(embedding_dim, len(derivative_orders))
        
        if n < embedding_dim:
            print(f"Warning: Signal length {n} is too short for embedding dimension {embedding_dim}")
            return np.array([])
        
        # 计算各阶导数
        derivatives = []
        for order in derivative_orders:
            if order == 0:
                deriv = window.copy()
            else:
                deriv = self.compute_derivatives(window, order)
            if smoothing and order > 0:
                deriv = self.smooth_signal(deriv)
            derivatives.append(deriv)
        
        # 构建嵌入点云
        point_cloud = np.zeros((n, embedding_dim))
        for i in range(embedding_dim):
            if i < len(derivatives):
                point_cloud[:, i] = derivatives[i]
        
        if show:
            self._visualize_derivative_embedding(window, point_cloud, derivative_orders[:embedding_dim])
        
        return point_cloud
    
    def compute_adaptive_derivative_embedding(self, window, max_dim=5, 
                                            adaptive_selection=True, show=False):
        """自适应导数嵌入，自动选择最优的导数组合"""
        window = np.asarray(window)
        
        if not adaptive_selection:
            return self.compute_derivative_embedding(window, embedding_dim=max_dim, show=show)
        
        best_combination = None
        best_score = -np.inf
        
        for dim in range(2, min(max_dim + 1, 4)):
            for orders in self._generate_derivative_combinations(dim):
                try:
                    point_cloud = self.compute_derivative_embedding(
                        window, embedding_dim=dim, derivative_orders=orders, show=False
                    )
                    
                    if len(point_cloud) > 0:
                        score = self._evaluate_embedding_quality(point_cloud)
                        if score > best_score:
                            best_score = score
                            best_combination = (point_cloud, orders)
                            
                except Exception as e:
                    continue
        
        if best_combination is None:
            return self.compute_derivative_embedding(window, embedding_dim=2, show=show)
        
        point_cloud, orders = best_combination
        
        if show:
            self._visualize_derivative_embedding(window, point_cloud, orders)
        
        return point_cloud
    
    def _generate_derivative_combinations(self, dim):
        """生成导数阶数的组合"""
        combinations = []
        for i in range(dim):
            combo = list(range(i + 1))
            if len(combo) == dim:
                combinations.append(combo)
        
        special_combinations = [
            [0, 1],      # 原始信号 + 一阶导数
            [0, 1, 2],   # 原始信号 + 一阶导数 + 二阶导数
            [1, 2],      # 一阶导数 + 二阶导数
            [0, 2],      # 原始信号 + 二阶导数
        ]
        
        for combo in special_combinations:
            if len(combo) == dim and combo not in combinations:
                combinations.append(combo)
        
        return combinations
    
    def _evaluate_embedding_quality(self, point_cloud):
        """评估嵌入质量"""
        if len(point_cloud) < 10:
            return -np.inf
        
        try:
            pca = PCA(n_components=min(3, point_cloud.shape[1]))
            pca.fit(point_cloud)
            explained_variance = pca.explained_variance_ratio_
            
            from sklearn.neighbors import NearestNeighbors
            nbrs = NearestNeighbors(n_neighbors=2).fit(point_cloud)
            distances, _ = nbrs.kneighbors(point_cloud)
            avg_distance = np.mean(distances[:, 1])
            
            std_distances = np.std(distances[:, 1])
            uniformity = 1.0 / (1.0 + std_distances / avg_distance)
            
            volume_score = np.sum(explained_variance)
            density_score = 1.0 / (1.0 + avg_distance)
            
            total_score = volume_score * density_score * uniformity
            
            return total_score
            
        except Exception as e:
            return -np.inf
    
    def _visualize_derivative_embedding(self, window, point_cloud, derivative_orders):
        """可视化导数嵌入结果"""
        try:
            dim = point_cloud.shape[1]
            
            if dim == 2:
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
                
                ax1.plot(window, 'b-', linewidth=1.5, alpha=0.8)
                ax1.set_title('Original Time Series', fontsize=12, fontweight='bold')
                ax1.set_xlabel('Time')
                ax1.set_ylabel('Amplitude')
                ax1.grid(True, alpha=0.3)
                
                scatter = ax2.scatter(point_cloud[:, 0], point_cloud[:, 1], 
                                    c=np.arange(len(point_cloud)), cmap='viridis', 
                                    alpha=0.7, s=20)
                ax2.set_title(f'Derivative Embedding (Orders: {derivative_orders})', 
                            fontsize=12, fontweight='bold')
                ax2.set_xlabel(f'Order {derivative_orders[0]}')
                ax2.set_ylabel(f'Order {derivative_orders[1]}')
                ax2.grid(True, alpha=0.3)
                plt.colorbar(scatter, ax=ax2, label='Time Index')
                
            elif dim == 3:
                fig = plt.figure(figsize=(15, 5))
                
                ax1 = fig.add_subplot(1, 3, 1)
                ax1.plot(window, 'b-', linewidth=1.5, alpha=0.8)
                ax1.set_title('Original Time Series', fontsize=12, fontweight='bold')
                ax1.set_xlabel('Time')
                ax1.set_ylabel('Amplitude')
                ax1.grid(True, alpha=0.3)
                
                ax2 = fig.add_subplot(1, 3, 2, projection='3d')
                scatter = ax2.scatter(point_cloud[:, 0], point_cloud[:, 1], point_cloud[:, 2],
                                    c=np.arange(len(point_cloud)), cmap='viridis', 
                                    alpha=0.7, s=20)
                ax2.set_title(f'3D Derivative Embedding\n(Orders: {derivative_orders})', 
                            fontsize=12, fontweight='bold')
                ax2.set_xlabel(f'Order {derivative_orders[0]}')
                ax2.set_ylabel(f'Order {derivative_orders[1]}')
                ax2.set_zlabel(f'Order {derivative_orders[2]}')
                
                ax3 = fig.add_subplot(1, 3, 3)
                pca = PCA(n_components=2)
                pca_cloud = pca.fit_transform(point_cloud)
                scatter = ax3.scatter(pca_cloud[:, 0], pca_cloud[:, 1],
                                    c=np.arange(len(point_cloud)), cmap='viridis', 
                                    alpha=0.7, s=20)
                ax3.set_title(f'PCA Projection\n(Explained: {pca.explained_variance_ratio_[0]:.1%}, {pca.explained_variance_ratio_[1]:.1%})', 
                            fontsize=12, fontweight='bold')
                ax3.set_xlabel('PC1')
                ax3.set_ylabel('PC2')
                ax3.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.show()
            
        except Exception as e:
            print(f"Visualization failed: {e}")
    
    def compute_phase_space_embedding(self, window, embedding_dim=3, delay=1, show=False):
        """计算相空间嵌入（传统延迟嵌入的替代方案）"""
        window = np.asarray(window)
        n = len(window)
        
        if n < embedding_dim * delay:
            return np.array([])
        
        # 计算相空间嵌入
        point_cloud = np.zeros((n - (embedding_dim - 1) * delay, embedding_dim))
        for i in range(embedding_dim):
            start_idx = i * delay
            end_idx = n - (embedding_dim - 1) * delay + start_idx
            point_cloud[:, i] = window[start_idx:end_idx]
        
        if show:
            self._visualize_phase_space_embedding(window, point_cloud, embedding_dim, delay)
        
        return point_cloud
    
    def _visualize_phase_space_embedding(self, window, point_cloud, embedding_dim, delay):
        """可视化相空间嵌入"""
        try:
            if embedding_dim == 2:
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
                
                ax1.plot(window, 'b-', linewidth=1.5, alpha=0.8)
                ax1.set_title('Original Time Series', fontsize=12, fontweight='bold')
                ax1.set_xlabel('Time')
                ax1.set_ylabel('Amplitude')
                ax1.grid(True, alpha=0.3)
                
                scatter = ax2.scatter(point_cloud[:, 0], point_cloud[:, 1], 
                                    c=np.arange(len(point_cloud)), cmap='viridis', 
                                    alpha=0.7, s=20)
                ax2.set_title(f'Phase Space Embedding (d={embedding_dim}, τ={delay})', 
                            fontsize=12, fontweight='bold')
                ax2.set_xlabel('x(t)')
                ax2.set_ylabel('x(t+τ)')
                ax2.grid(True, alpha=0.3)
                plt.colorbar(scatter, ax=ax2, label='Time Index')
                
            elif embedding_dim == 3:
                fig = plt.figure(figsize=(15, 5))
                
                ax1 = fig.add_subplot(1, 3, 1)
                ax1.plot(window, 'b-', linewidth=1.5, alpha=0.8)
                ax1.set_title('Original Time Series', fontsize=12, fontweight='bold')
                ax1.set_xlabel('Time')
                ax1.set_ylabel('Amplitude')
                ax1.grid(True, alpha=0.3)
                
                ax2 = fig.add_subplot(1, 3, 2, projection='3d')
                scatter = ax2.scatter(point_cloud[:, 0], point_cloud[:, 1], point_cloud[:, 2],
                                    c=np.arange(len(point_cloud)), cmap='viridis', 
                                    alpha=0.7, s=20)
                ax2.set_title(f'3D Phase Space Embedding\n(d={embedding_dim}, τ={delay})', 
                            fontsize=12, fontweight='bold')
                ax2.set_xlabel('x(t)')
                ax2.set_ylabel('x(t+τ)')
                ax2.set_zlabel('x(t+2τ)')
                
                ax3 = fig.add_subplot(1, 3, 3)
                pca = PCA(n_components=2)
                pca_cloud = pca.fit_transform(point_cloud)
                scatter = ax3.scatter(pca_cloud[:, 0], pca_cloud[:, 1],
                                    c=np.arange(len(point_cloud)), cmap='viridis', 
                                    alpha=0.7, s=20)
                ax3.set_title(f'PCA Projection\n(Explained: {pca.explained_variance_ratio_[0]:.1%}, {pca.explained_variance_ratio_[1]:.1%})', 
                            fontsize=12, fontweight='bold')
                ax3.set_xlabel('PC1')
                ax3.set_ylabel('PC2')
                ax3.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.show()
            
        except Exception as e:
            print(f"Phase space visualization failed: {e}")


def compare_embedding_methods(timeseries, show_comparison=True):
    """比较不同嵌入方法的性能"""
    processor = DerivativeEmbeddingProcessor()
    
    # 传统延迟嵌入
    delay_embedding = processor.compute_phase_space_embedding(
        timeseries, embedding_dim=3, delay=5, show=False
    )
    
    # 导数嵌入
    derivative_embedding = processor.compute_derivative_embedding(
        timeseries, embedding_dim=3, derivative_orders=[0, 1, 2], show=False
    )
    
    # 自适应导数嵌入
    adaptive_embedding = processor.compute_adaptive_derivative_embedding(
        timeseries, max_dim=3, show=False
    )
    
    # 评估质量
    delay_score = processor._evaluate_embedding_quality(delay_embedding)
    derivative_score = processor._evaluate_embedding_quality(derivative_embedding)
    adaptive_score = processor._evaluate_embedding_quality(adaptive_embedding)
    
    results = {
        'delay_embedding': {
            'point_cloud': delay_embedding,
            'score': delay_score,
            'method': 'Phase Space Embedding'
        },
        'derivative_embedding': {
            'point_cloud': derivative_embedding,
            'score': derivative_score,
            'method': 'Derivative Embedding'
        },
        'adaptive_embedding': {
            'point_cloud': adaptive_embedding,
            'score': adaptive_score,
            'method': 'Adaptive Derivative Embedding'
        }
    }
    
    if show_comparison:
        _visualize_embedding_comparison(timeseries, results)
    
    return results


def _visualize_embedding_comparison(timeseries, results):
    """可视化不同嵌入方法的比较"""
    try:
        fig = plt.figure(figsize=(18, 12))
        
        # 原始信号
        ax1 = fig.add_subplot(2, 3, 1)
        ax1.plot(timeseries, 'b-', linewidth=1.5, alpha=0.8)
        ax1.set_title('Original Time Series', fontsize=12, fontweight='bold')
        ax1.set_xlabel('Time')
        ax1.set_ylabel('Amplitude')
        ax1.grid(True, alpha=0.3)
        
        # 延迟嵌入
        ax2 = fig.add_subplot(2, 3, 2, projection='3d')
        delay_pc = results['delay_embedding']['point_cloud']
        if len(delay_pc) > 0:
            scatter = ax2.scatter(delay_pc[:, 0], delay_pc[:, 1], delay_pc[:, 2],
                                c=np.arange(len(delay_pc)), cmap='viridis', 
                                alpha=0.7, s=20)
        ax2.set_title(f"Delay Embedding\nScore: {results['delay_embedding']['score']:.3f}", 
                    fontsize=12, fontweight='bold')
        ax2.set_xlabel('x(t)')
        ax2.set_ylabel('x(t+τ)')
        ax2.set_zlabel('x(t+2τ)')
        
        # 导数嵌入
        ax3 = fig.add_subplot(2, 3, 3, projection='3d')
        deriv_pc = results['derivative_embedding']['point_cloud']
        if len(deriv_pc) > 0:
            scatter = ax3.scatter(deriv_pc[:, 0], deriv_pc[:, 1], deriv_pc[:, 2],
                                c=np.arange(len(deriv_pc)), cmap='viridis', 
                                alpha=0.7, s=20)
        ax3.set_title(f"Derivative Embedding\nScore: {results['derivative_embedding']['score']:.3f}", 
                    fontsize=12, fontweight='bold')
        ax3.set_xlabel('Signal')
        ax3.set_ylabel('1st Derivative')
        ax3.set_zlabel('2nd Derivative')
        
        # 自适应导数嵌入
        ax4 = fig.add_subplot(2, 3, 4, projection='3d')
        adapt_pc = results['adaptive_embedding']['point_cloud']
        if len(adapt_pc) > 0:
            scatter = ax4.scatter(adapt_pc[:, 0], adapt_pc[:, 1], adapt_pc[:, 2],
                                c=np.arange(len(adapt_pc)), cmap='viridis', 
                                alpha=0.7, s=20)
        ax4.set_title(f"Adaptive Derivative Embedding\nScore: {results['adaptive_embedding']['score']:.3f}", 
                    fontsize=12, fontweight='bold')
        ax4.set_xlabel('Component 1')
        ax4.set_ylabel('Component 2')
        ax4.set_zlabel('Component 3')
        
        # 质量评分比较
        ax5 = fig.add_subplot(2, 3, 5)
        methods = list(results.keys())
        scores = [results[method]['score'] for method in methods]
        colors = ['skyblue', 'lightcoral', 'lightgreen']
        
        bars = ax5.bar(methods, scores, color=colors, alpha=0.8)
        ax5.set_title('Embedding Quality Comparison', fontsize=12, fontweight='bold')
        ax5.set_ylabel('Quality Score')
        ax5.grid(True, alpha=0.3)
        
        for bar, score in zip(bars, scores):
            height = bar.get_height()
            ax5.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{score:.3f}', ha='center', va='bottom', fontweight='bold')
        
        best_method = methods[np.argmax(scores)]
        best_score = max(scores)
        ax5.text(0.5, 0.95, f'Best: {best_method.replace("_", " ").title()}\nScore: {best_score:.3f}', 
                transform=ax5.transAxes, ha='center', va='top',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7),
                fontweight='bold')
        
        plt.tight_layout()
        plt.show()
        
    except Exception as e:
        print(f"Comparison visualization failed: {e}") 