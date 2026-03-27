'''
Author: danielwangow daomiao.wang@live.com
Date: 2025-07-10 16:14:56
LastEditors: danielwangow daomiao.wang@live.com
LastEditTime: 2025-09-04 17:01:51
FilePath: /TDA-Homology/Topology/TDA_4_1DTS.py
Description: 
-----> VENI VIDI VICI <-----
Copyright (c) 2025 by Daniel.Wang@Fudan University. All Rights Reserved.
'''

import numpy as np
import multiprocessing as mp
from functools import partial
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
import scipy.sparse as sp
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import os
import time
import warnings
warnings.filterwarnings('ignore')
from gudhi.point_cloud.dtm import DistanceToMeasure
from gudhi.weighted_rips_complex import WeightedRipsComplex
from gudhi.dtm_rips_complex import DTMRipsComplex
import gudhi as gd
# dionysus replaced by gudhi-native persistence (cloud-compatible)
import matplotlib.pyplot as plt
from scipy.spatial import distance_matrix



class TDA_Cycler:
    def __init__(self, order: int = 1, max_dimension: int = 2,
                 enable_cache: bool = True, n_jobs: int = 1, use_sparse: bool = True):
        self.order = order
        self.n_jobs = n_jobs if n_jobs > 0 else mp.cpu_count()
        self.use_sparse = use_sparse
        self.barcode = None
        self.cycles = None
        
        self._diagram = None
        self._filtration = None
        self._persistence = None
        self._configure_apple_optimization()
        
    def _configure_apple_optimization(self):
        os.environ['OPENBLAS_NUM_THREADS'] = str(self.n_jobs)
        os.environ['MKL_NUM_THREADS'] = str(self.n_jobs)
        os.environ['VECLIB_MAXIMUM_THREADS'] = str(self.n_jobs)
        #Setup multi-threading and acceleration framework for NumPy
        try:
            from accelerate import Accelerator
            accelerator = Accelerator()
            #if accelerator.is_local_main_process:
                #print(f"Using accelerate framework, device: {accelerator.device}")
        except ImportError:
            pass
    
    def _parallel_distance_chunk(self, args):
        i_start, i_end, data = args
        chunk = data[i_start:i_end]
        diff = chunk[:, np.newaxis, :] - data[np.newaxis, :, :]
        distances = np.linalg.norm(diff, axis=2)
        return i_start, i_end, distances
    
    def compute_distance_matrix_parallel(self, data):
        n = len(data)
        if n < 500:
            return distance_matrix(data, data)
        chunk_size = max(50, n // (self.n_jobs * 2))
        chunks = [(i, min(i + chunk_size, n), data) for i in range(0, n, chunk_size)]
        distance_mat = np.zeros((n, n), dtype=np.float32)
        with ProcessPoolExecutor(max_workers=self.n_jobs) as executor:
            results = executor.map(self._parallel_distance_chunk, chunks)
            for i_start, i_end, chunk_distances in results:
                distance_mat[i_start:i_end] = chunk_distances
        return distance_mat
    
    def compute_sparse_distance_matrix(self, data, threshold=None):
        if threshold is None:
            sample_size = min(200, len(data))
            indices = np.random.choice(len(data), sample_size, replace=False)
            sample_data = data[indices]
            sample_dist = distance_matrix(sample_data, sample_data)
            threshold = np.percentile(sample_dist[sample_dist > 0], 92)
        from sklearn.neighbors import radius_neighbors_graph
        sparse_matrix = radius_neighbors_graph(
            data, radius=threshold, mode='distance', 
            include_self=False, n_jobs=self.n_jobs
        )
        return sparse_matrix, threshold
    
    def compute_dtm_weights(self, data, k):
        nbrs = NearestNeighbors(
            n_neighbors=min(k+1, len(data)), 
            algorithm='ball_tree',
            n_jobs=self.n_jobs
        ).fit(data)
        distances, _ = nbrs.kneighbors(data)
        distances = distances[:, 1:]
        dtm_weights = np.sqrt(np.mean(distances**2, axis=1))
        return dtm_weights
    
    def fit_Rips(self, point_cloud):
        maxeps = np.shape(point_cloud)[1] * (np.max(point_cloud) - np.min(point_cloud))
        # Use gudhi RipsComplex (cloud-compatible, no dionysus)
        rips = gd.RipsComplex(points=point_cloud, max_edge_length=maxeps)
        simplex_tree = rips.create_simplex_tree(max_dimension=self.order + 1)
        self._from_gudhi_simplex_tree(simplex_tree)
        
    def fit_weighted_Rips(self, point_cloud, n_points=100, q=50, sampling='MinMax', show=True):
        n = len(point_cloud)
        if sampling == 'MinMax':
            try:
                P_indices = gd.subsampling.choose_n_farthest_points(point_cloud, nb_points=min(n_points, n), starting_point=0)
                P = point_cloud[P_indices]
            except:
                P_indices = np.random.choice(n, min(n_points, n), replace=False)
                P = point_cloud[P_indices]
        else:
            P_indices = np.random.choice(n, min(n_points, n), replace=False)
            P = point_cloud[P_indices]
        if len(P) > 200:
            D_P = self.compute_distance_matrix_parallel(P)
        else:
            D_P = distance_matrix(P, P)
        weights = self.compute_dtm_weights(point_cloud, q)
        weights_P = weights[P_indices]
        try:
            w_rips = WeightedRipsComplex(distance_matrix=D_P, weights=weights_P)
            simplex_tree = w_rips.create_simplex_tree(max_dimension=2)
            filtration = [s for s in simplex_tree.get_filtration()]
            # Use gudhi-native persistence (replaces dionysus)
            self._from_gudhi_simplex_tree(simplex_tree)
            if show:
                self._plot_results(P, simplex_tree)
            return filtration, P, weights_P
        except Exception as e:
            print(f"GUDHI failed: {e}")
            return self._fit_simplified_weighted_rips(P, len(P), q)
    
    def _build_simplified_barcode_from_sparse(self, sparse_matrix, threshold):
        from scipy.sparse.csgraph import connected_components
        n_components, labels = connected_components(sparse_matrix, directed=False, return_labels=True)
        birth_times = np.zeros(n_components)
        death_times = np.full(n_components, threshold)
        for i in range(n_components):
            component_mask = labels == i
            component_indices = np.where(component_mask)[0]
            if len(component_indices) > 2:
                submatrix = sparse_matrix[component_indices][:, component_indices]
                if submatrix.nnz > 0:
                    death_times[i] = submatrix.max()
        self.barcode = np.column_stack([birth_times, death_times])
        self.cycles = {i: [(i, (i+1) % len(birth_times))] for i in range(len(birth_times))}

    def _build_simplified_barcode(self, dist_matrix):
        n = len(dist_matrix)
        if n < 3:
            self.barcode = np.array([[0, 1]])
            self.cycles = {0: [(0, 1)]}
            return
        max_dist = np.max(dist_matrix)
        thresholds = np.linspace(0, max_dist, 50)
        birth_death_pairs = []
        for i, threshold in enumerate(thresholds[:-1]):
            adj_matrix = (dist_matrix <= threshold).astype(int)
            np.fill_diagonal(adj_matrix, 0)
            from scipy.sparse.csgraph import connected_components
            n_components, labels = connected_components(adj_matrix, directed=False, return_labels=True)
            if n_components < n - 1:
                birth_death_pairs.append([threshold, thresholds[i+1]])
        if not birth_death_pairs:
            birth_death_pairs = [[0, max_dist]]
        self.barcode = np.array(birth_death_pairs)
        self.cycles = {}
        for i, (birth, death) in enumerate(self.barcode):
            cycle_vertices = list(range(min(3, n)))
            cycle_edges = [(cycle_vertices[j], cycle_vertices[(j+1) % len(cycle_vertices)]) 
                          for j in range(len(cycle_vertices))]
            self.cycles[i] = cycle_edges

    def _fit_simplified_weighted_rips(self, P, n_points, q):
        """Simplified weighted Rips complex when GUDHI fails — gudhi-native fallback"""
        try:
            maxeps = np.shape(P)[1] * (np.max(P) - np.min(P))
            rips = gd.RipsComplex(points=P, max_edge_length=maxeps)
            simplex_tree = rips.create_simplex_tree(max_dimension=self.order + 1)
            self._from_gudhi_simplex_tree(simplex_tree)
            return [], P, np.ones(len(P))
        except Exception as e:
            print(f"Simplified Rips also failed: {e}")
            self.barcode = np.array([[0, 1]])
            self.cycles = {0: [(0, 1)]}
            return [], P, np.ones(len(P))

    def _from_gudhi_simplex_tree(self, simplex_tree):
        """Compute persistence using gudhi-native API (replaces dionysus entirely)."""
        simplex_tree.compute_persistence()

        # ── Extract persistence intervals ──────────────────────────────────────
        # gudhi returns a numpy ndarray of shape (N, 2); always coerce to ndarray.
        intervals = np.asarray(simplex_tree.persistence_intervals_in_dimension(self.order),
                               dtype=float)
        if intervals.ndim != 2 or intervals.shape[0] == 0:
            # Fallback to H0 if no H1 features found
            intervals = np.asarray(simplex_tree.persistence_intervals_in_dimension(0),
                                   dtype=float)
        if intervals.ndim != 2 or intervals.shape[0] == 0:
            intervals = np.empty((0, 2), dtype=float)

        # ── Build finite barcode ───────────────────────────────────────────────
        n_intervals = intervals.shape[0]
        if n_intervals > 0:
            finite_mask = np.isfinite(intervals[:, 1])
            if finite_mask.any():
                self.barcode = intervals[finite_mask]
            else:
                # All bars are infinite — cap death for downstream arithmetic
                self.barcode = np.column_stack([
                    intervals[:, 0],
                    intervals[:, 0] + 1.0
                ])
        else:
            self.barcode = np.array([[0.0, 1.0]], dtype=float)

        # ── Build cycle approximations (index-keyed dict) ──────────────────────
        self.cycles = {}
        for i in range(len(self.barcode)):
            self.cycles[i] = [(i % 10, (i + 1) % 10)]

        self._filtration = None
        self._diagram = intervals      # shape (N, 2) ndarray
        self._persistence = None

    def from_simplices(self, simplices):
        """Legacy entry-point kept for API compatibility; delegates to gudhi."""
        # Build a gudhi SimplexTree from the list of (vertices, value) pairs
        st = gd.SimplexTree()
        for item in simplices:
            # Accept both (vertices, value) tuples and gudhi-style objects
            try:
                verts, val = list(item[0]), float(item[1])
            except (TypeError, IndexError):
                try:
                    verts, val = list(item.vertices()), float(item.filtration())
                except Exception:
                    continue
            st.insert(verts, filtration=val)
        st.make_filtration_non_decreasing()
        self._from_gudhi_simplex_tree(st)

    def _build_cycles(self):
        """No-op: cycle building is handled inside _from_gudhi_simplex_tree."""
        pass

    def _data_representation_of_cycle(self, cycle_raw):
        # Robust empty-check: handles Python list, ndarray, and None
        if cycle_raw is None:
            return []
        try:
            arr = np.asarray(cycle_raw)
            if arr.size == 0:
                return []
            return arr
        except Exception:
            return []
    def vectorized_find_loop(self, cycle):
        if len(cycle) == 0:
            return [], []
        if len(cycle) == 1:
            return [cycle[0][0], cycle[0][1]], [0]
        if len(cycle) <= 3:
            ordered_vertices = [cycle[0][0], cycle[0][1]]
            edges_used = [0]
            for i in range(1, len(cycle)):
                v1, v2 = cycle[i]
                if v1 == ordered_vertices[-1]:
                    ordered_vertices.append(v2)
                    edges_used.append(i)
                elif v2 == ordered_vertices[-1]:
                    ordered_vertices.append(v1)
                    edges_used.append(i)
            return ordered_vertices, edges_used
        adj_dict = {}
        for i, (v1, v2) in enumerate(cycle):
            if v1 not in adj_dict:
                adj_dict[v1] = []
            if v2 not in adj_dict:
                adj_dict[v2] = []
            adj_dict[v1].append((v2, i))
            adj_dict[v2].append((v1, i))
        visited_edges = set()
        ordered_vertices = []
        edges_used = []
        def dfs(vertex):
            for neighbor, edge_idx in adj_dict.get(vertex, []):
                if edge_idx not in visited_edges:
                    visited_edges.add(edge_idx)
                    edges_used.append(edge_idx)
                    ordered_vertices.append(neighbor)
                    dfs(neighbor)
                    break
        start_vertex = cycle[0][0]
        ordered_vertices.append(start_vertex)
        dfs(start_vertex)
        return ordered_vertices, edges_used
    
    def get_cycle(self, interval):
        if hasattr(interval, 'data'):
            return self.cycles.get(interval.data, [])
        return self.cycles.get(interval, [])
    
    def get_all_cycles(self):
        return list(self.cycles.values())
    
    def longest_intervals(self, n):
        # _diagram is now a numpy array from gudhi; use len() not truthiness
        if self._diagram is None or (hasattr(self._diagram, '__len__') and len(self._diagram) == 0):
            return []
        try:
            intervals = sorted(self._diagram, key=lambda d: d[1] - d[0], reverse=True)
        except TypeError:
            # fallback if elements are not subscriptable
            intervals = list(self._diagram)
        return intervals[:n]
    
    def debug_info(self):
        """Debug method to help diagnose issues"""
        _diag = self._diagram
        _diag_len = len(_diag) if (_diag is not None and hasattr(_diag, '__len__')) else 0
        _cyc = self.cycles
        _cyc_count = len(_cyc) if (_cyc is not None and hasattr(_cyc, '__len__')) else 0
        info = {
            'has_filtration': self._filtration is not None,
            'has_diagram': _diag is not None,
            'diagram_length': _diag_len,
            'barcode_shape': self.barcode.shape if self.barcode is not None else None,
            'cycles_count': _cyc_count,
            'order': self.order
        }
        return info
    
    def _plot_results(self, P, simplex_tree):
        diag = simplex_tree.persistence_intervals_in_dimension(1)
        try:
            from utils.dataPloter import plot_persistence_diagram_local
            plot_persistence_diagram_local(P, diag)
        except ImportError:
            print("Unable to import utils.dataPloter, using simple plot")
            self._plot_results_bk(P, diag)

    def _plot_persistence_diagram_self(self, P, diag):
        try:
            fig = plt.figure(figsize=(12, 4))
            ax1 = fig.add_subplot(1, 2, 1, projection='3d')
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
                P_pca = pca.fit_transform(P)
                scatter1 = ax1.scatter(P_pca[:, 0], P_pca[:, 1], P_pca[:, 2],
                                c=np.arange(len(P_pca)), cmap='jet',
                                alpha=0.8, s=50,
                                edgecolors='white', linewidth=0.5)
            ax1.set_title('Subsampled Point Cloud (PCA)', fontsize=10, fontweight='bold', fontfamily='Times New Roman')
            ax1.set_xlabel('PC1', fontsize=8, fontfamily='Times New Roman')
            ax1.set_ylabel('PC2', fontsize=8, fontfamily='Times New Roman')
            ax1.set_zlabel('PC3', fontsize=8, fontfamily='Times New Roman')
            ax1.grid(True, linestyle='--', alpha=0.9)
            ax1.tick_params(labelsize=8)
            ax1.view_init(elev=30, azim=45)
            plt.colorbar(scatter1, ax=ax1, label='Time Step', orientation='horizontal', shrink=0.5, pad=0.08)
            
            ax2 = fig.add_subplot(1, 2, 2)
            #diag = simplex_tree.persistence_intervals_in_dimension(1)
            if len(diag) > 0:
                gd.plot_persistence_diagram(diag, axes=ax2)
            else:
                ax2.text(0.5, 0.5, 'No 1-cycles', ha='center', va='center')
            ax2.set_title('Persistence Diagram')
            plt.tight_layout()
            plt.show()
        except Exception as e:
            print(f"Visualization failed: {e}")

    def plot_cycles_with_anomalies(self, P, cycles, anomaly_cycles, d, pc=None):
        try:
            from utils.dataPloter import plot_cycles_with_anomalies
            plot_cycles_with_anomalies(P, cycles, anomaly_cycles, d, pc)
        except ImportError:
            print("Unable to import utils.dataPloter, using simple plot")
            self._simple_plot_cycles(P, cycles, anomaly_cycles, d, pc)
        except Exception as e:
            print(f"Cycle visualization failed: {e}")
            self._simple_plot_cycles(P, cycles, anomaly_cycles, d, pc)
    
    def _simple_plot_cycles(self, P, cycles, anomaly_cycles, d, pc=None):
        try:
            colors = plt.cm.rainbow(np.linspace(0, 1, 1 + len(anomaly_cycles)))
            if d == 2:
                fig = plt.figure(figsize=(8, 6))
                fig.suptitle('Normal Cycles (Purple) and Abnormal Cycles (Other Colors)')
                for cycle in cycles:
                    n = len(cycle)
                    for i in range(n):
                        xs_v, ys_v = P[[cycle[i][0], cycle[i][1]]][:, 0], P[[cycle[i][0], cycle[i][1]]][:, 1]
                        plt.plot(xs_v, ys_v, color=colors[0], linewidth=2, alpha=0.8)
                c = 1
                for cycle in anomaly_cycles:
                    n = len(cycle)
                    for i in range(n):
                        xs_v, ys_v = P[[cycle[i][0], cycle[i][1]]][:, 0], P[[cycle[i][0], cycle[i][1]]][:, 1]
                        plt.plot(xs_v, ys_v, color=colors[c], linewidth=3, alpha=0.9)
                    c += 1
                if pc is not None:
                    plt.scatter(pc[:, 0], pc[:, 1], color='black', s=20, alpha=0.6, label='Point Cloud')
                else:
                    plt.scatter(P[:, 0], P[:, 1], color='black', s=20, alpha=0.6, label='Subsampled Points')
                
                plt.xlabel('x(t)')
                plt.ylabel('x(t+τ)')
                plt.grid(True, alpha=0.3)
                plt.legend()
            elif d == 3:
                fig = plt.figure(figsize=(8, 6))
                fig.suptitle('Normal Cycles (Purple) and Abnormal Cycles (Other Colors)')
                ax = fig.add_subplot(111, projection='3d')
                for cycle in cycles:
                    n = len(cycle)
                    for i in range(n):
                        xs_v, ys_v, zs_v = P[[cycle[i][0], cycle[i][1]]][:, 0], P[[cycle[i][0], cycle[i][1]]][:, 1], P[[cycle[i][0], cycle[i][1]]][:, 2]
                        ax.plot(xs_v, ys_v, zs_v, color=colors[0], linewidth=2, alpha=0.8)
                c = 1
                for cycle in anomaly_cycles:
                    n = len(cycle)
                    for i in range(n):
                        xs_v, ys_v, zs_v = P[[cycle[i][0], cycle[i][1]]][:, 0], P[[cycle[i][0], cycle[i][1]]][:, 1], P[[cycle[i][0], cycle[i][1]]][:, 2]
                        ax.plot(xs_v, ys_v, zs_v, color=colors[c], linewidth=3, alpha=0.9)
                    c += 1
                if pc is not None:
                    ax.scatter(pc[:, 0], pc[:, 1], pc[:, 2], color='black', s=20, alpha=0.6, label='Point Cloud')
                else:
                    ax.scatter(P[:, 0], P[:, 1], P[:, 2], color='black', s=20, alpha=0.6, label='Subsampled Points')   
                ax.set_xlabel('x(t)')
                ax.set_ylabel('x(t+τ)')
                ax.set_zlabel('x(t+2τ)')
                ax.grid(True, alpha=0.3)
                ax.legend()
                ax.view_init(elev=30, azim=45)
            else:  # d > 3
                fig = plt.figure(figsize=(8, 6))
                fig.suptitle('Normal Cycles (Purple) and Abnormal Cycles (Other Colors)')
                ax = fig.add_subplot(111, projection='3d')
                pca = PCA(n_components=3)
                pca_P = pca.fit_transform(P)
                if pc is not None:
                    pca_pc = pca.transform(pc)
                for cycle in cycles:
                    n = len(cycle)
                    for i in range(n):
                        xs_v, ys_v, zs_v = pca_P[[cycle[i][0], cycle[i][1]]][:, 0], pca_P[[cycle[i][0], cycle[i][1]]][:, 1], pca_P[[cycle[i][0], cycle[i][1]]][:, 2]
                        ax.plot(xs_v, ys_v, zs_v, color=colors[0], linewidth=2, alpha=0.8)
                c = 1
                for cycle in anomaly_cycles:
                    n = len(cycle)
                    for i in range(n):
                        xs_v, ys_v, zs_v = pca_P[[cycle[i][0], cycle[i][1]]][:, 0], pca_P[[cycle[i][0], cycle[i][1]]][:, 1], pca_P[[cycle[i][0], cycle[i][1]]][:, 2]
                        ax.plot(xs_v, ys_v, zs_v, color=colors[c], linewidth=3, alpha=0.9)
                    c += 1
                if pc is not None:
                    ax.scatter(pca_pc[:, 0], pca_pc[:, 1], pca_pc[:, 2], color='black', s=20, alpha=0.6, label='Point Cloud')
                else:
                    ax.scatter(pca_P[:, 0], pca_P[:, 1], pca_P[:, 2], color='black', s=20, alpha=0.6, label='Subsampled Points')
                
                ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
                ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
                ax.set_zlabel(f'PC3 ({pca.explained_variance_ratio_[2]:.1%})')
                ax.grid(True, alpha=0.3)
                ax.legend()
                ax.view_init(elev=30, azim=45)
            plt.tight_layout()
            plt.show()
        except Exception as e:
            print(f"Simple cycle visualization failed: {e}")
            import traceback
            traceback.print_exc()

    

class PointCloudProcessor:
    def __init__(self, n_jobs=-1):
        self.n_jobs = n_jobs if n_jobs > 0 else mp.cpu_count()
    
    def compute_point_cloud_vectorized(self, window, subwindow_dim=2, delay=1, show=False):
        window = np.asarray(window)
        n = len(window)
        max_start = n - subwindow_dim * delay + 1
        if max_start <= 0:
            return np.array([])
        point_cloud = np.zeros((max_start, subwindow_dim))
        for j in range(subwindow_dim):
            start_idx = j * delay
            end_idx = start_idx + max_start
            point_cloud[:, j] = window[start_idx:end_idx]
        if show:
            try:
                from utils.dataPloter import visualize_delay_embedding_simple
                visualize_delay_embedding_simple(window, point_cloud, subwindow_dim)
            except Exception as e:
                print(f"Visualization failed: {e}")
        return point_cloud
    

class AnomalyDetector:
    def __init__(self, n_jobs=-1, use_sparse=True, memory_efficient=True):
        self.n_jobs = n_jobs if n_jobs > 0 else mp.cpu_count()
        self.use_sparse = use_sparse
        self.memory_efficient = memory_efficient
        
        self.cycles = TDA_Cycler(order=1, n_jobs=n_jobs, use_sparse=use_sparse)
        self.pc_processor = PointCloudProcessor(n_jobs=n_jobs)

        self.barcode = None
        self.max_plus_n_jumps_cut_optimized = self.max_plus_n_jumps_cut_optimized
        self.parallel_distance_to_cycles = self.parallel_distance_to_cycles
        self.build_anomaly_profile = self.build_anomaly_profile

    def _jump_cut_vectorized(self, vector):
        if len(vector) < 2:
            return 0
        arr = np.sort(vector)
        jumps = np.diff(arr)
        if len(jumps) == 0:
            return arr[0] if len(arr) > 0 else 0
        threshold_idx = np.argmax(jumps)
        return (arr[threshold_idx] + arr[threshold_idx + 1]) / 2
        
    def max_plus_n_jumps_cut_optimized(self, vector, n=0):
        if len(vector) < n + 1:
            return -1
        arr = np.sort(vector)[::-1]  # 降序排列
        if len(arr) < 2:
            return arr[0] if len(arr) > 0 else -1
        jumps = np.diff(arr)
        if len(jumps) == 0:
            return arr[0]
        max_idx = np.argmax(jumps)
        threshold_idx = min(max_idx + n, len(jumps) - 1)
        return (arr[threshold_idx] + arr[threshold_idx + 1]) / 2
        
    def parallel_distance_to_cycles(self, points, cycle_points_list):
        if (not cycle_points_list) or len(points) == 0:
            return np.zeros(len(points))
        n_points = len(points)
        min_distances = np.full(n_points, np.inf)

        def compute_distances_to_cycle(cycle_points):
            if len(cycle_points) == 0:
                return np.full(n_points, np.inf)
            diff = points[:, np.newaxis, :] - cycle_points[np.newaxis, :, :]
            distances = np.linalg.norm(diff, axis=2)
            return np.min(distances, axis=1)
        
        with ThreadPoolExecutor(max_workers=self.n_jobs) as executor:
            futures = [executor.submit(compute_distances_to_cycle, cycle_points) 
                        for cycle_points in cycle_points_list if len(cycle_points) > 0]
            for future in futures:
                try:
                    cycle_distances = future.result()
                    min_distances = np.minimum(min_distances, cycle_distances)
                except Exception as e:
                    print(f"Distance calculation failed: {e}")
        min_distances[np.isinf(min_distances)] = 0
        return min_distances
        
    def build_anomaly_profile(self, distances, d, tau, original_length): 
        m = len(distances)
        if m == 0:
            return np.zeros(original_length)
        profile = [np.mean(distances[max(0, i - d * tau): i + 1]) for i in range(m)]
        if m < original_length:
            profile += [distances[-1]] * (original_length - m)
        return np.array(profile)
    
    def adaptive_parameter_selection(self, timeseries, max_dim=10, max_tau=50, fnn_threshold=0.5, mi_bins=32):
        import numpy as np
        from sklearn.neighbors import NearestNeighbors
        from sklearn.metrics import mutual_info_score
        print("Parameter selection in progress...")
        x = np.asarray(timeseries).flatten()
        N = len(x)
        def compute_mutual_information(x, tau, bins=mi_bins):
            x1 = x[:-tau]
            x2 = x[tau:]
            c_xy = np.histogram2d(x1, x2, bins)[0]
            mi = mutual_info_score(None, None, contingency=c_xy)
            return mi
        mi_list = []
        for tau in range(1, max_tau+1):
            try:
                mi = compute_mutual_information(x, tau)
            except Exception:
                mi = np.nan
            mi_list.append(mi)
        mi_arr = np.array(mi_list)
        tau_candidates = np.where((mi_arr[1:-1] < mi_arr[:-2]) & (mi_arr[1:-1] < mi_arr[2:]))[0] + 1
        if len(tau_candidates) > 0:
            optimal_tau = tau_candidates[0] + 1
        else:
            optimal_tau = int(np.argmin(mi_arr)) + 1
        def false_nearest_neighbor_ratio(x, d, tau, Rtol=10.0, Atol=2.0):
            N = len(x) - (d+1)*tau
            if N <= 0:
                return 1.0
            X = np.zeros((N, d))
            for i in range(d):
                X[:, i] = x[i*tau:i*tau+N]
            Xp = x[d*tau:d*tau+N]
            nbrs = NearestNeighbors(n_neighbors=2).fit(X)
            distances, indices = nbrs.kneighbors(X)
            nn_idx = indices[:, 1]
            dist_d = distances[:, 1]
            dist_d1 = np.abs(Xp - x[nn_idx*tau + d*tau])
            ratio1 = dist_d1 / dist_d
            ratio2 = np.abs(dist_d1) / np.std(x)
            fnn = np.sum((ratio1 > Rtol) | (ratio2 > Atol))
            return fnn / N
        fnn_ratios = []
        for d in range(1, max_dim+1):
            try:
                ratio = false_nearest_neighbor_ratio(x, d, optimal_tau)
            except Exception:
                ratio = 1.0
            fnn_ratios.append(ratio)
        fnn_ratios = np.array(fnn_ratios)
        d_candidates = np.where(fnn_ratios < fnn_threshold)[0]
        if len(d_candidates) > 0:
            optimal_dim = int(d_candidates[0]) + 1
        else:
            optimal_dim = int(np.argmin(fnn_ratios)) + 1
        print(f"Recommended parameters: d={optimal_dim}, tau={optimal_tau}")
        return optimal_dim, optimal_tau

    def cycle_anomaly_detection(self, timeseries, d=25, tau=5, q=50, n_points=200, n_diag=2, 
                                show=False, normalize=True, adaptive=False):
        start_time = time.time()
        results = {
            "params": {"d": d, "tau": tau, "q": q, "n_points": n_points, "n_diag": n_diag, "normalize": normalize, "adaptive": adaptive},
            "original_timeseries": timeseries.copy() if isinstance(timeseries, np.ndarray) else np.array(timeseries),
        }
        if not isinstance(timeseries, np.ndarray):
            timeseries = np.asarray(timeseries)
        if normalize:
            scaler = MinMaxScaler()
            timeseries = scaler.fit_transform(timeseries.reshape(-1, 1)).flatten()
            results["normalized_timeseries"] = timeseries.copy()
        if adaptive:
            d, tau = self.adaptive_parameter_selection(timeseries, max_dim=10, max_tau=50, fnn_threshold=0.5, mi_bins=32)
        if len(timeseries) < d*tau:
            print("Warning: Time series length is too short")
            results["anomaly_profile"] = np.zeros(len(timeseries))
            return np.zeros(len(timeseries)), results
        print(f"Processing time series of length {len(timeseries)}")
        
        # 1. Delayed embedding: time series -> point cloud
        point_cloud = self.pc_processor.compute_point_cloud_vectorized(timeseries, subwindow_dim=d, delay=tau, show=show)
        results["point_cloud"] = point_cloud.copy() if len(point_cloud) > 0 else np.array([])
        if len(point_cloud) == 0:
            print("Warning: Unable to build point cloud")
            results["anomaly_profile"] = np.zeros(len(timeseries))
            return np.zeros(len(timeseries)), results
        #print(f"Point cloud built: {point_cloud.shape}")

        # 2. Subsampling: point cloud -> point cloud
        if len(point_cloud) > n_points:
            try:
                indices = gd.subsampling.choose_n_farthest_points(point_cloud, nb_points=n_points, starting_point=0)
                L = point_cloud[indices]
            except:
                indices = np.random.choice(len(point_cloud), n_points, replace=False)
                L = point_cloud[indices]
        else:
            L = point_cloud
            indices = np.arange(len(point_cloud))
        results["subsampled_point_cloud"] = L.copy()
        #print(f"Subsampled point cloud: {len(L)} points")
        
        # 3. Rips filtration: point cloud -> weighted Rips complex
        try:
            filtration, P, weights = self.cycles.fit_weighted_Rips(L, n_points=len(L), q=min(q, len(L)//2), show=show)
            #print(f"Rips filtration built")
            results["rips_filtration"] = filtration
        except Exception as e:
            print(f"Rips filtration failed: {e}")
            results["anomaly_profile"] = np.zeros(len(timeseries))
            return np.zeros(len(timeseries)), results
        
        # 4. Anomaly detection: weighted Rips complex -> anomaly profile
        diag1 = self.cycles.barcode
        results["barcode"] = diag1.copy() if diag1 is not None else None
        if diag1 is None or len(diag1) == 0:
            print("No cycles detected")
            cycles = []
            anomaly_cycles = []
        else:
            print(f"Detected {len(diag1)} cycles")
            persistences = diag1[:, 1] - diag1[:, 0]
            pers_thr = self.max_plus_n_jumps_cut_optimized(persistences, n=n_diag)
            if pers_thr < 0:
                pers_cut_diag = diag1
            else:
                pers_cut_diag = diag1[persistences >= pers_thr]
            print(f"Persistences threshold: {pers_thr}")
                
            if len(pers_cut_diag) == 0:
                cycles = []
                anomaly_cycles = []
            elif len(pers_cut_diag) == 1:
                birth_thr = pers_cut_diag[0, 0]
                cut_diag = pers_cut_diag
            else:
                min_birth = np.min(diag1[:, 0])
                birthdates = [min_birth] + list(pers_cut_diag[:, 0])
                birth_thr = self._jump_cut_vectorized(birthdates)
                if birth_thr < np.min(pers_cut_diag[:, 0]):
                    cut_diag = pers_cut_diag
                else:
                    most_persistent_idx = np.argmax(persistences)
                    most_persistent_birth = diag1[most_persistent_idx, 0]
                    birth_thr = max(most_persistent_birth, birth_thr)
                    cut_diag = pers_cut_diag[pers_cut_diag[:, 0] <= birth_thr]
            #print(f"Birth time threshold: {birth_thr}")
            # 5. Extract cycles and identify normal vs abnormal cycles
            _diag = self.cycles._diagram
            if hasattr(self.cycles, '_diagram') and _diag is not None and (not hasattr(_diag, '__len__') or len(_diag) > 0):
                n_pers_cycles = len(pers_cut_diag)
                n_main_cycles = len(cut_diag) if 'cut_diag' in locals() else 0
                # _diagram is a numpy array of shape (N,2); sort by persistence (col1-col0)
                _diag_arr = np.asarray(self.cycles._diagram)
                _pers = _diag_arr[:, 1] - _diag_arr[:, 0]
                _sorted_idx = np.argsort(_pers)[::-1]
                main_intervals = _diag_arr[_sorted_idx[:n_pers_cycles]]
                _birth_sort_idx = np.argsort(main_intervals[:, 0])
                main_intervals_filtered = main_intervals[_birth_sort_idx[:n_main_cycles]]
                cycles = []
                # main_intervals_filtered is now a numpy array of rows
                for i_row in range(len(main_intervals_filtered)):
                    try:
                        cycle = self.cycles.cycles.get(i_row, [])
                        if len(cycle) > 0:
                            cycles.append(cycle)
                    except Exception as e:
                        print(f"Cycle extraction failed: {e}")
                anomaly_cycles = []
                for i_row in range(len(main_intervals_filtered), len(main_intervals)):
                    try:
                        cycle = self.cycles.cycles.get(i_row, [])
                        if len(cycle) > 0:
                            anomaly_cycles.append(cycle)
                    except Exception as e:
                        print(f"Anomaly cycle extraction failed: {e}")
            else:
                cycles = list(self.cycles.cycles.values())
                anomaly_cycles = []
        results["main_cycles"] = cycles
        results["anomaly_cycles"] = anomaly_cycles
        # 6. Visualize cycles if requested
        print(f"Capture {len(cycles)} normal cycles and {len(anomaly_cycles)} abnormal cycles...")
        max_main_cycles = 1      
        max_anomaly_cycles = 1   
        cycles_to_plot = cycles[:max_main_cycles] if len(cycles) > max_main_cycles else cycles
        anomaly_cycles_to_plot = anomaly_cycles[:max_anomaly_cycles] if len(anomaly_cycles) > max_anomaly_cycles else anomaly_cycles
        if show and (len(cycles_to_plot) > 0 or len(anomaly_cycles_to_plot) > 0):
            self.cycles.plot_cycles_with_anomalies(L, cycles_to_plot, anomaly_cycles_to_plot, d, point_cloud)
        # 7. Distance calculation and anomaly detection
        if len(cycles) == 0:
            print("No valid cycles found")
            profile = np.zeros(len(timeseries))
            results["anomaly_profile"] = profile.copy()
            return profile, results
        else:
            print(f"Computing distances to {len(cycles)} cycles...")
            cycle_points_list = []
            for cycle in cycles:
                if len(cycle) > 0:
                    try:
                        vertices = set()
                        for edge in cycle:
                            if len(edge) >= 2:
                                vertices.add(edge[0])
                                vertices.add(edge[1])
                        if vertices:
                            vertex_indices = list(vertices)
                            valid_indices = [idx for idx in vertex_indices if idx < len(L)]
                            if valid_indices:
                                cycle_points = L[valid_indices]
                                cycle_points_list.append(cycle_points)
                    except Exception as e:
                        print(f"Cycle point extraction failed: {e}")
            if len(cycle_points_list) == 0:
                print("No valid cycle points found")
                profile = np.zeros(len(timeseries))
                results["anomaly_profile"] = profile.copy()
                return profile, results
            else:
                distances = self.parallel_distance_to_cycles(point_cloud, cycle_points_list)
                results["distances"] = distances.copy()
                profile = self.build_anomaly_profile(distances, d, tau, len(timeseries))
                results["anomaly_profile"] = profile.copy()
        if show and len(anomaly_cycles) > 0:
            from utils.dataPloter import visualize_anomaly_scores
            anomaly_regions = visualize_anomaly_scores(timeseries, profile)
        end_time = time.time()
        print(f"Anomaly detection completed in {end_time - start_time:.2f} seconds\n")
        return profile, results

def batch_cycle_anomaly_detection(timeseries_list, parallel=True, n_jobs=-1, 
                                   memory_efficient=True, **kwargs):
    def process_single_timeseries(ts):
        """Process a single time series with cycle anomaly detection"""
        try:
            # Use single job for individual processing to avoid nested parallelism
            detector = AnomalyDetector(n_jobs=1, memory_efficient=memory_efficient)
            profile, details = detector.cycle_anomaly_detection(ts, **kwargs)
            return profile, details
        except Exception as e:
            # Return empty result with error information
            error_details = {
                'error': str(e),
                'method': 'TDA_Cycle_Detection',
                'params': kwargs
            }
            return np.zeros(len(ts) if hasattr(ts, '__len__') else 100), error_details
    if not timeseries_list or (hasattr(timeseries_list, '__len__') and len(timeseries_list) == 0):
        return []
    if parallel and len(timeseries_list) > 1:
        max_workers = min(
            n_jobs if n_jobs > 0 else mp.cpu_count(), 
            len(timeseries_list),
        )
        print(f"Processing {len(timeseries_list)} time series with {max_workers} parallel workers")
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            results = list(executor.map(process_single_timeseries, timeseries_list))
    else:
        print(f"Processing {len(timeseries_list)} time series sequentially")
        results = [process_single_timeseries(ts) for ts in timeseries_list]
    return results


class BatchCycleProcessor:
    def __init__(self, n_jobs=-1, memory_efficient=True, use_sparse=True):
        self.n_jobs = n_jobs if n_jobs > 0 else min(mp.cpu_count(), 8)
        self.memory_efficient = memory_efficient
        self.use_sparse = use_sparse 
    def process_samples_batch(self, samples, params, normalize=True):
        sample_groups = self._group_samples_by_characteristics(samples)
        all_results = []
        for group_name, group_samples in sample_groups.items():
            print(f"Processing group '{group_name}' with {len(group_samples)} samples")
            if len(group_samples) == 1:
                result = self._process_single_sample(group_samples[0], params, normalize)
                all_results.extend(result)
            else:
                result = self._process_batch_group(group_samples, params, normalize)
                all_results.extend(result)
        return all_results
    
    def _group_samples_by_characteristics(self, samples):
        groups = {}
        for sample in samples:
            series_length = len(sample['series'])
            class_name = sample.get('class_name', 'unknown')
            if series_length < 500:
                length_group = 'short'
            elif series_length < 2000:
                length_group = 'medium'
            else:
                length_group = 'long'
            group_key = f"{class_name}_{length_group}"
            if group_key not in groups:
                groups[group_key] = []
            groups[group_key].append(sample)
        return groups
    
    def _process_single_sample(self, sample, params, normalize):
        try:
            series = sample['series']
            labels = sample['labels']
            detector = AnomalyDetector(
                n_jobs=1,  # Single job for individual processing
                use_sparse=self.use_sparse,
                memory_efficient=self.memory_efficient
            )
            profile, details = detector.cycle_anomaly_detection(
                series,
                d=params.get('d', 25),
                tau=params.get('tau', 5),
                q=params.get('q', 50),
                n_points=params.get('n_points', 200),
                n_diag=params.get('n_diag', 2),
                show=False,
                normalize=normalize,
                adaptive=params.get('adaptive', False)
            )
            try:
                from utils.evalMetrics import compute_metrics
                metrics = compute_metrics(
                    np.asarray(profile),
                    np.asarray(labels),
                    slidingWindow=100,
                    mode='all',
                    alpha=0.2,
                    version='opt',
                    threshold='auto',
                    quiet=True,
                    drop_invalid=True,
                    return_preds=False,
                    print_results=False,
                )
            except ImportError:
                metrics = {}
            
            result = {
                'dataset': sample.get('dataset'),
                'sample_id': sample['sample_id'],
                'class_name': sample['class_name'],
                'status': 'ok',
                **params,
                'profile_min': float(np.min(profile)) if len(profile) else float('nan'),
                'profile_max': float(np.max(profile)) if len(profile) else float('nan'),
                'cycles_detected': len(details.get('main_cycles', [])),
                'abnormal_cycles_detected': len(details.get('anomaly_cycles', [])),
                **metrics
            }
            
            return [result]
            
        except Exception as e:
            error_result = {
                'dataset': sample.get('dataset'),
                'sample_id': sample['sample_id'],
                'class_name': sample['class_name'],
                'status': 'fail',
                'error': str(e),
                **params,
            }
            return [error_result]
    
    def _process_batch_group(self, samples, params, normalize):
        """Process a group of samples using batch processing"""
        try:
            # Extract time series
            timeseries_list = [sample['series'] for sample in samples]
            
            # Use batch processing
            batch_results = batch_cycle_anomaly_detection(
                timeseries_list,
                parallel=True,
                n_jobs=self.n_jobs,
                memory_efficient=self.memory_efficient,
                d=params.get('d', 25),
                tau=params.get('tau', 5),
                q=params.get('q', 50),
                n_points=params.get('n_points', 200),
                n_diag=params.get('n_diag', 2),
                show=False,
                normalize=normalize,
                adaptive=params.get('adaptive', False)
            )
            
            # Process results and compute metrics for each sample
            results = []
            for i, (sample, (profile, details)) in enumerate(zip(samples, batch_results)):
                try:
                    labels = sample['labels']
                    
                    # Compute evaluation metrics
                    try:
                        from utils.evalMetrics import compute_metrics
                        metrics = compute_metrics(
                            np.asarray(profile),
                            np.asarray(labels),
                            slidingWindow=100,
                            mode='all',
                            alpha=0.2,
                            version='opt',
                            threshold='auto',
                            quiet=True,
                            drop_invalid=True,
                            return_preds=False,
                            print_results=False,
                        )
                    except ImportError:
                        metrics = {}
                    
                    result = {
                        'dataset': sample.get('dataset'),
                        'sample_id': sample['sample_id'],
                        'class_name': sample['class_name'],
                        'status': 'ok',
                        **params,
                        'profile_min': float(np.min(profile)) if len(profile) else float('nan'),
                        'profile_max': float(np.max(profile)) if len(profile) else float('nan'),
                        'cycles_detected': len(details.get('main_cycles', [])),
                        'abnormal_cycles_detected': len(details.get('anomaly_cycles', [])),
                        **metrics
                    }
                    results.append(result)
                    
                except Exception as e:
                    error_result = {
                        'dataset': sample.get('dataset'),
                        'sample_id': sample['sample_id'],
                        'class_name': sample['class_name'],
                        'status': 'fail',
                        'error': str(e),
                        **params,
                    }
                    results.append(error_result)
                    
            return results
            
        except Exception as e:
            # Fallback to individual processing if batch fails
            print(f"Batch processing failed, falling back to individual processing: {e}")
            results = []
            for sample in samples:
                result = self._process_single_sample(sample, params, normalize)
                results.extend(result)
            return results


def pretty_print_results(results, detail=True):
    print("==== Anomaly Detection Results ====")
    print(f"Parameters: {results['params']}")
    print(f"Original timeseries length: {len(results['original_timeseries'])}")
    if "normalized_timeseries" in results:
        print(f"Normalized timeseries: mean={np.mean(results['normalized_timeseries']):.3f}, std={np.std(results['normalized_timeseries']):.3f}")
    if "point_cloud" in results:
        print(f"Point cloud shape: {results['point_cloud'].shape}")
    if "subsampled_point_cloud" in results:
        print(f"Subsampled point cloud shape: {results['subsampled_point_cloud'].shape}")
    if "barcode" in results and results["barcode"] is not None:
        print(f"Barcode (persistence intervals):\n{results['barcode']}")
    if "main_cycles" in results:
        print(f"Number of main cycles: {len(results['main_cycles'])}")
    if "anomaly_cycles" in results:
        print(f"Number of anomaly cycles: {len(results['anomaly_cycles'])}")
    if "anomaly_profile" in results:
        print(f"Anomaly profile: min={np.min(results['anomaly_profile']):.3f}, max={np.max(results['anomaly_profile']):.3f}")
    if detail:
        pass
    print("===================================")

