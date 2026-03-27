# Author: danielwangow daomiao.wang@live.com
# Date: 2025-08-06 15:33:00
# LastEditors: danielwangow daomiao.wang@live.com
# LastEditTime: 2025-08-06 15:33:00
# FilePath: /TDA-Homology/Topology/CardiovascularMetrics.py
# Description: 
# -----> VENI VIDI VICI <-----
# Copyright (c) 2025 by Daniel.Wang@Fudan University. All Rights Reserved.

import numpy as np
import time
import json
from sklearn.decomposition import PCA
from scipy.signal import find_peaks
import warnings
warnings.filterwarnings('ignore')


class CardiovascularMetricsExtractor:
    def __init__(self):
        self.metrics = {}
    
    def extract_all_metrics(self, results_dict, processing_time, sampling_rate=100):
        all_metrics = {
            'analysis_timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'analysis_version': 'CardioDetector_TDA_v1.0 by Daniel.Wang@xxx University'
        }

        # Basic Signal Metrics
        # 1. Signal properties: signal_frequency_hz, signal_duration_second
        # 2. Signal numeric features: mean_amplitude, std_amplitude, amplitude_range
        # 3. Frequency domain features: dominant_frequency_hz
        # 4. Signal quality assessment: signal_quality
        if 'original_timeseries' in results_dict:
            all_metrics['basic_signal_metrics'] = self._extract_basic_signal_metrics(
                results_dict['original_timeseries'], sampling_rate)
        
        # Topology Metrics
        # 1. Cycle properties: total_cycles, normal_cycles, anomaly_cycles, anomaly_ratio
        # 2. Persistence: mean_persistence, max_persistence, persistence_std
        if 'barcode' in results_dict and 'main_cycles' in results_dict:
            all_metrics['topology_metrics'] = self._extract_topology_metrics(
                results_dict['barcode'],
                results_dict['main_cycles'],
                results_dict.get('anomaly_cycles', []))
        
        # Anomaly Metrics
        # 1. Anomaly profile: mean_anomaly_score, max_anomaly_score, anomaly_score_std
        # 2. Anomaly coverage: anomaly_coverage_90p_percent, anomaly_coverage_95p_percent
        # 3. Anomaly peak count: anomaly_peak_count
        if 'anomaly_profile' in results_dict:
            all_metrics['anomaly_metrics'] = self._extract_anomaly_metrics(
                results_dict['anomaly_profile'])
        
        # Cardiovascular Metrics
        # 1. Estimated heart rate: estimated_heart_rate_bpm
        # 2. Severity distribution: mild_percent, moderate_percent, severe_percent
        if 'original_timeseries' in results_dict and 'anomaly_profile' in results_dict:
            all_metrics['cardiovascular_metrics'] = self._extract_cardiovascular_metrics(
                results_dict['original_timeseries'],
                results_dict['anomaly_profile'],
                sampling_rate
            )
        # Computational Metrics
        all_metrics['computational_metrics'] = {
            'processing_time_seconds': processing_time
        }
        
        # Summary Report
        all_metrics['summary'] = self._generate_summary(all_metrics)
        
        return all_metrics
    
    def _extract_basic_signal_metrics(self, timeseries, sampling_rate):
        metrics = {
            'signal_frequency_hz': sampling_rate,
            'signal_duration_seconds': len(timeseries) / sampling_rate,
            'mean_amplitude': float(np.mean(timeseries)),
            'std_amplitude': float(np.std(timeseries)),
            'amplitude_range': float(np.max(timeseries) - np.min(timeseries)),
            'signal_to_noise_ratio': float(10 * np.log10(np.sum(np.square(timeseries)) / (len(timeseries) * np.var(timeseries)))) if np.var(timeseries) > 0 else 0
        }
        fft = np.fft.fft(timeseries)
        freqs = np.fft.fftfreq(len(timeseries), 1/sampling_rate)
        positive_freqs = freqs[freqs > 0]
        positive_fft = np.abs(fft[freqs > 0])
        if len(positive_freqs) > 0:
            dominant_freq_idx = np.argmax(positive_fft)
            metrics['dominant_frequency_hz'] = float(positive_freqs[dominant_freq_idx])
        return metrics
    
    def _extract_topology_metrics(self, barcode, cycles, anomaly_cycles):
        if barcode is None or len(barcode) == 0:
            return {
                'total_cycles': 0,
                'normal_cycles': 0,
                'anomaly_cycles': 0,
                'anomaly_ratio': 0.0
                }
        persistences = barcode[:, 1] - barcode[:, 0]
        return {
            'total_cycles': len(barcode),
            'normal_cycles': len(cycles),
            'anomaly_cycles': len(anomaly_cycles),
            'anomaly_ratio': len(anomaly_cycles) / max(len(barcode), 1),
            'mean_persistence': float(np.mean(persistences)),
            'max_persistence': float(np.max(persistences)),
            'persistence_std': float(np.std(persistences))
        }
    
    def _extract_anomaly_metrics(self, anomaly_profile):
        metrics = {
            'mean_anomaly_score': float(np.mean(anomaly_profile)),
            'max_anomaly_score': float(np.max(anomaly_profile)),
            'anomaly_score_std': float(np.std(anomaly_profile))
            }
        threshold_90 = np.percentile(anomaly_profile, 90)
        threshold_95 = np.percentile(anomaly_profile, 95)
        high_anomaly_90 = anomaly_profile > threshold_90
        high_anomaly_95 = anomaly_profile > threshold_95
        metrics['anomaly_coverage_90p_percent'] = float(100 * np.sum(high_anomaly_90) / len(anomaly_profile))
        metrics['anomaly_coverage_95p_percent'] = float(100 * np.sum(high_anomaly_95) / len(anomaly_profile))
        peaks, _ = find_peaks(anomaly_profile, height=np.percentile(anomaly_profile, 75))
        metrics['anomaly_peak_count'] = len(peaks)
        return metrics
    
    def _extract_cardiovascular_metrics(self, timeseries, anomaly_profile, sampling_rate):
        metrics = {}
        if len(anomaly_profile) > sampling_rate * 5:
            peaks, _ = find_peaks(timeseries, height=np.percentile(timeseries, 75))
            metrics['estimated_heart_rate_bpm'] = len(peaks) * 60 * sampling_rate / len(timeseries)
        if len(anomaly_profile) > 0:
            severity_thresholds = {
                'mild': np.percentile(anomaly_profile, 75),
                'moderate': np.percentile(anomaly_profile, 90),
                'severe': np.percentile(anomaly_profile, 95)
            }
            metrics['severity_distribution'] = {
                'mild_percent': float(100 * np.sum(anomaly_profile > severity_thresholds['mild']) / len(anomaly_profile)),
                'moderate_percent': float(100 * np.sum(anomaly_profile > severity_thresholds['moderate']) / len(anomaly_profile)),
                'severe_percent': float(100 * np.sum(anomaly_profile > severity_thresholds['severe']) / len(anomaly_profile))
            }
        return metrics
    
    def _generate_summary(self, metrics):
        summary = {
            'signal_quality': self._assess_signal_quality(metrics),
            'anomaly_level': self._assess_anomaly_level(metrics),
            'cardiovascular_status': self._assess_cardiovascular_status(metrics),
            'recommendations': self._generate_recommendations(metrics)
        }
        return summary
    
    def _assess_signal_quality(self, metrics):
        if 'basic_signal_metrics' not in metrics:
            return "Failed to assess signal quality (No basic signal metrics detected...)"
        
        snr = metrics['basic_signal_metrics'].get('signal_to_noise_ratio', 0)
        if snr > 5:
            return "Good for analysis (SNR > 5)"
        elif snr > 2:
            return "Fair for analysis (SNR > 2)"
        else:
            return "Poor for analysis (SNR < 2)"
    
    def _assess_anomaly_level(self, metrics):
        if 'anomaly_metrics' not in metrics:
            return "Failed to assess anomaly level (No anomaly detected...)"
        coverage = metrics['anomaly_metrics'].get('anomaly_coverage_95p_percent', 0)
        if coverage < 5:
            return "Normal (Anomaly coverage < 5%)"
        elif coverage < 20:
            return "Mild anomaly (Anomaly coverage < 20%)"
        elif coverage < 40:
            return "Moderate anomaly (Anomaly coverage < 40%)"
        else:
            return "Severe anomaly (Anomaly coverage >= 40%)"
    
    def _assess_cardiovascular_status(self, metrics):
        if 'cardiovascular_metrics' not in metrics:
            return "Failed to assess cardiovascular status (No cardiovascular metrics detected...)"
        severe_percent = metrics['cardiovascular_metrics'].get('severity_distribution', {}).get('severe_percent', 0)
        if severe_percent < 1:
            return "Healthy (Severe anomaly < 1%)"
        elif severe_percent < 5:
            return "Mild anomaly (Severe anomaly < 5%)"
        elif severe_percent < 15:
            return "Moderate anomaly (Severe anomaly < 15%)"
        else:
            return "Severe anomaly (Severe anomaly >= 15%)"
    

    def _generate_recommendations(self, metrics):
        recommendations = []
        
        anomaly_level = self._assess_anomaly_level(metrics)
        if anomaly_level == "Severe anomaly (Anomaly coverage >= 40%)":
            recommendations.append("Immediate cardiovascular check is recommended")
        elif anomaly_level == "Moderate anomaly (Anomaly coverage < 40%)":
            recommendations.append("Regular monitoring of cardiovascular status is recommended and consider")
        elif anomaly_level == "Mild anomaly (Anomaly coverage < 20%)":
            recommendations.append("Regular monitoring of cardiovascular status is recommended")
        
        signal_quality = self._assess_signal_quality(metrics)
        if signal_quality == "Poor for analysis (SNR < 2)":
            recommendations.append("Improve signal acquisition quality")
        if not recommendations:
            recommendations.append("Current detection result can be considered as normal")
        return recommendations
    
    def save_metrics_to_json(self, metrics, filepath):
        def make_json_serializable(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, dict):
                return {key: make_json_serializable(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [make_json_serializable(item) for item in obj]
            else:
                return obj
        serializable_metrics = make_json_serializable(metrics)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(serializable_metrics, f, indent=2, ensure_ascii=False)
        print(f"Metrics saved to {filepath}")


def create_enhanced_detector():
    from Topology.TDA_4_1DTS_results import AnomalyDetector
    class EnhancedAnomalyDetector(AnomalyDetector):
        def __init__(self, n_jobs=-1, use_sparse=True, memory_efficient=True):
            super().__init__(n_jobs, use_sparse, memory_efficient)
            self.metrics_extractor = CardiovascularMetricsExtractor()
        def detect_with_metrics(self, timeseries, d=25, tau=5, q=50, n_points=200, n_diag=2, 
                              show=False, normalize=True, adaptive=False, 
                              save_metrics_path=None, sampling_rate=1000):
            start_time = time.time()
            profile, results = self.cycle_anomaly_detection(
                timeseries, d, tau, q, n_points, n_diag, show, normalize, adaptive
            )
            processing_time = time.time() - start_time
            metrics = self.metrics_extractor.extract_all_metrics(
                results, processing_time, sampling_rate
            )
            if save_metrics_path:
                self.metrics_extractor.save_metrics_to_json(metrics, save_metrics_path)
            results['metrics'] = metrics
            return profile, results, metrics
    return EnhancedAnomalyDetector


def print_enhanced_results(results):
    print("==== Cardiovascular Anomaly Detection Results (using TDA_4_1DTS) ====")
    if "metrics" in results:
        metrics = results['metrics']
        if 'summary' in metrics:
            summary = metrics['summary']
            print(f"Signal Quality: {summary['signal_quality']}")
            print(f"Anomaly Level: {summary['anomaly_level']}")
            print(f"Cardiovascular Status: {summary['cardiovascular_status']}")
            print(f"Recommendations: {', '.join(summary['recommendations'])}")
        if 'basic_signal_metrics' in metrics:
            basic = metrics['basic_signal_metrics']
            print(f"\nSignal Length: {basic['signal_frequency_hz']} HZ")
            print(f"Signal Duration: {basic['signal_duration_seconds']:.2f} seconds")
            print(f"Signal-to-Noise Ratio: {basic['signal_to_noise_ratio']:.3f}")
        if 'topology_metrics' in metrics:
            topo = metrics['topology_metrics']
            print(f"\nTotal Cycles: {topo['total_cycles']}")
            print(f"Anomaly Ratio: {topo['anomaly_ratio']:.3f}")
        if 'anomaly_metrics' in metrics:
            anomaly = metrics['anomaly_metrics']
            print(f"\nMean Anomaly Score: {anomaly['mean_anomaly_score']:.3f}")
            print(f"Anomaly Coverage (95%): {anomaly['anomaly_coverage_95p_percent']:.2f}%")
        if 'cardiovascular_metrics' in metrics:
            cv = metrics['cardiovascular_metrics']
            if 'estimated_heart_rate_bpm' in cv:
                print(f"\nEstimated Heart Rate: {cv['estimated_heart_rate_bpm']:.1f} bpm")
    print("=============================") 