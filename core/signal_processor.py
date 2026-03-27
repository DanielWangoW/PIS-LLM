'''
Author: danielwangow daomiao.wang@live.com
Description: TDA-based signal processing pipeline adapter for ProEngOpt.
             Bridges the TDA-Homology library with the Streamlit web application.
             Handles CSV ingestion, TDA analysis, metric extraction, and plot generation.
             v2.1: Removed SNR metric; raised plot DPI to 220.
-----> VENI VIDI VICI <-----
Copyright (c) 2025 by Daniel.Wang@Fudan University. All Rights Reserved.
'''

import sys
import time
import uuid
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for server-side rendering
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Any, Optional
from sklearn.decomposition import PCA

warnings.filterwarnings("ignore")

# ── Inject TDA-Homology source path ───────────────────────────────────────────
_TDA_SRC = Path(__file__).parent / "tda_lib"
if str(_TDA_SRC) not in sys.path:
    sys.path.insert(0, str(_TDA_SRC))

# ── Lazy imports with graceful fallback ───────────────────────────────────────
try:
    from TDA_4_1DTS import AnomalyDetector
    from CardiovascularMetrics import CardiovascularMetricsExtractor
    _TDA_AVAILABLE = True
    _TDA_IMPORT_ERROR = ""
except ImportError as e:
    _TDA_AVAILABLE = False
    _TDA_IMPORT_ERROR = str(e)

# ── Plot DPI constant ──────────────────────────────────────────────────────────
_PLOT_DPI = 220


# ══════════════════════════════════════════════════════════════════════════════
# Public API
# ══════════════════════════════════════════════════════════════════════════════

class SignalProcessingResult:
    """Typed container for all analysis outputs."""

    def __init__(self):
        self.success: bool = False
        self.error_message: str = ""
        self.timeseries: Optional[np.ndarray] = None
        self.anomaly_profile: Optional[np.ndarray] = None
        self.metrics: Dict[str, Any] = {}
        self.processing_time: float = 0.0
        self.plot_glance_path: Optional[str] = None
        self.plot_tda_path: Optional[str] = None
        self.sampling_rate: int = 100

    @property
    def summary(self) -> Dict[str, Any]:
        return self.metrics.get("summary", {})

    @property
    def basic_signal(self) -> Dict[str, Any]:
        # Strip SNR from basic signal metrics before exposing
        raw = dict(self.metrics.get("basic_signal_metrics", {}))
        raw.pop("signal_to_noise_ratio", None)
        return raw

    @property
    def topology(self) -> Dict[str, Any]:
        return self.metrics.get("topology_metrics", {})

    @property
    def anomaly(self) -> Dict[str, Any]:
        return self.metrics.get("anomaly_metrics", {})

    @property
    def cardiovascular(self) -> Dict[str, Any]:
        return self.metrics.get("cardiovascular_metrics", {})

    def to_context_dict(self) -> Dict[str, Any]:
        """Return a flat dict of key metrics for LLM conversation context."""
        basic    = self.basic_signal
        topo     = self.topology
        anomaly  = self.anomaly
        cardio   = self.cardiovascular
        severity = cardio.get("severity_distribution", {})
        summary  = self.summary
        return {
            "signal_frequency_hz":          basic.get("signal_frequency_hz", "N/A"),
            "signal_duration_seconds":      round(basic.get("signal_duration_seconds", 0), 1),
            "dominant_frequency_hz":        round(basic.get("dominant_frequency_hz", 0), 2),
            "total_cycles":                 topo.get("total_cycles", 0),
            "normal_cycles":                topo.get("normal_cycles", 0),
            "anomaly_cycles":               topo.get("anomaly_cycles", 0),
            "anomaly_ratio_pct":            round(topo.get("anomaly_ratio", 0) * 100, 1),
            "mean_persistence":             round(topo.get("mean_persistence", 0), 4),
            "mean_anomaly_score":           round(anomaly.get("mean_anomaly_score", 0), 4),
            "max_anomaly_score":            round(anomaly.get("max_anomaly_score", 0), 4),
            "anomaly_coverage_90p_percent": round(anomaly.get("anomaly_coverage_90p_percent", 0), 1),
            "anomaly_coverage_95p_percent": round(anomaly.get("anomaly_coverage_95p_percent", 0), 1),
            "anomaly_peak_count":           anomaly.get("anomaly_peak_count", 0),
            "estimated_heart_rate_bpm":     round(cardio.get("estimated_heart_rate_bpm", 0), 1),
            "mild_percent":                 round(severity.get("mild_percent", 0), 1),
            "moderate_percent":             round(severity.get("moderate_percent", 0), 1),
            "severe_percent":               round(severity.get("severe_percent", 0), 1),
            "anomaly_level":                summary.get("anomaly_level", "N/A"),
            "cardiovascular_status":        summary.get("cardiovascular_status", "N/A"),
            "recommendations":              summary.get("recommendations", []),
            "processing_time_s":            round(self.processing_time, 2),
            "analysis_version":             self.metrics.get("analysis_version", "CardioDetector_TDA_v2.1"),
        }


def process_signal_file(
    file_path: str,
    output_dir: str,
    tda_params: Dict[str, Any],
    sampling_rate: int = 100,
) -> SignalProcessingResult:
    """
    Full pipeline: CSV → TDA analysis → metrics extraction → plot generation.

    Parameters
    ----------
    file_path    : Path to the uploaded CSV/TXT signal file.
    output_dir   : Directory where output PNG plots will be saved.
    tda_params   : Dict with keys: d, tau, q, n_points, n_diag, normalize, adaptive.
    sampling_rate: Sampling frequency in Hz (default 100 Hz).

    Returns
    -------
    SignalProcessingResult with all analysis outputs populated.
    """
    result = SignalProcessingResult()
    result.sampling_rate = sampling_rate

    if not _TDA_AVAILABLE:
        result.error_message = (
            f"TDA library not available. Import error: {_TDA_IMPORT_ERROR}\n"
            "Please ensure gudhi, dionysus, and scikit-learn are installed."
        )
        return result

    # ── 1. Load signal data ────────────────────────────────────────────────────
    try:
        timeseries = _load_signal(file_path)
        result.timeseries = timeseries
    except Exception as e:
        result.error_message = f"Failed to load signal file: {e}"
        return result

    # ── 2. Run TDA anomaly detection ──────────────────────────────────────────
    try:
        t0 = time.time()
        detector = AnomalyDetector(n_jobs=1, use_sparse=True, memory_efficient=True)
        profile, tda_results = detector.cycle_anomaly_detection(
            timeseries,
            d=tda_params.get("d", 25),
            tau=tda_params.get("tau", 5),
            q=tda_params.get("q", 50),
            n_points=tda_params.get("n_points", 200),
            n_diag=tda_params.get("n_diag", 2),
            show=False,
            normalize=tda_params.get("normalize", True),
            adaptive=tda_params.get("adaptive", False),
        )
        processing_time = time.time() - t0
        result.anomaly_profile = profile
        result.processing_time = processing_time
    except Exception as e:
        result.error_message = f"TDA analysis failed: {e}"
        return result

    # ── 3. Extract cardiovascular metrics ─────────────────────────────────────
    try:
        extractor = CardiovascularMetricsExtractor()
        metrics = extractor.extract_all_metrics(
            tda_results, processing_time, sampling_rate=sampling_rate
        )
        result.metrics = metrics
    except Exception as e:
        result.error_message = f"Metrics extraction failed: {e}"
        return result

    # ── 4. Generate plots ──────────────────────────────────────────────────────
    session_id = uuid.uuid4().hex[:8]
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        glance_path = str(out_dir / f"glance_{session_id}.png")
        _plot_signal_and_anomaly(timeseries, profile, glance_path)
        result.plot_glance_path = glance_path
    except Exception:
        result.plot_glance_path = None

    try:
        tda_path = str(out_dir / f"tda_{session_id}.png")
        _plot_tda_topology(tda_results, tda_path)
        result.plot_tda_path = tda_path
    except Exception:
        result.plot_tda_path = None

    result.success = True
    return result


def build_llm_prompt(result: SignalProcessingResult, template_path: str) -> str:
    """
    Render the prompt template with real metrics from SignalProcessingResult.
    SNR is excluded from the data dict.
    """
    with open(template_path, "r", encoding="utf-8") as f:
        template = f.read()

    ctx = result.to_context_dict()
    severity_dist = result.cardiovascular.get("severity_distribution", {})

    data = {
        "signal_frequency_hz":          ctx["signal_frequency_hz"],
        "signal_duration_seconds":      ctx["signal_duration_seconds"],
        "dominant_frequency_hz":        ctx["dominant_frequency_hz"],
        "total_cycles":                 ctx["total_cycles"],
        "normal_cycles":                ctx["normal_cycles"],
        "anomaly_cycles":               ctx["anomaly_cycles"],
        "anomaly_ratio":                f"{ctx['anomaly_ratio_pct']}%",
        "mean_persistence":             ctx["mean_persistence"],
        "mean_anomaly_score":           ctx["mean_anomaly_score"],
        "max_anomaly_score":            ctx["max_anomaly_score"],
        "anomaly_coverage_90p_percent": f"{ctx['anomaly_coverage_90p_percent']}%",
        "anomaly_coverage_95p_percent": f"{ctx['anomaly_coverage_95p_percent']}%",
        "anomaly_peak_count":           ctx["anomaly_peak_count"],
        "estimated_heart_rate_bpm":     ctx["estimated_heart_rate_bpm"],
        "mild_percent":                 f"{ctx['mild_percent']}%",
        "moderate_percent":             f"{ctx['moderate_percent']}%",
        "severe_percent":               f"{ctx['severe_percent']}%",
        "anomaly_level":                ctx["anomaly_level"],
        "cardiovascular_status":        ctx["cardiovascular_status"],
        "signal_quality":               result.summary.get("signal_quality", "N/A"),
        "recommendations":              "; ".join(ctx["recommendations"]),
        "analysis_version":             ctx["analysis_version"],
        "analysis_timestamp":           result.metrics.get("analysis_timestamp", "N/A"),
        "processing_time":              ctx["processing_time_s"],
    }

    try:
        return template.format(**data)
    except KeyError:
        return template


# ══════════════════════════════════════════════════════════════════════════════
# Internal helpers
# ══════════════════════════════════════════════════════════════════════════════

def _load_signal(file_path: str) -> np.ndarray:
    """Load a 1-D signal from CSV or TXT. Handles various column layouts."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Signal file not found: {file_path}")

    try:
        df = pd.read_csv(file_path, header=None)
    except Exception:
        df = pd.read_csv(file_path, sep="\t", header=None)

    if df.empty:
        raise ValueError("Uploaded file is empty or unreadable.")

    for col in df.columns:
        series = pd.to_numeric(df[col], errors="coerce").dropna()
        if len(series) > 10:
            return series.values.astype(np.float64)

    raise ValueError(
        "No valid numeric column found in the uploaded file. "
        "Please ensure the file contains a single-column numeric time series."
    )


def _plot_signal_and_anomaly(
    timeseries: np.ndarray,
    anomaly_scores: np.ndarray,
    save_path: str,
):
    """Generate a 3-panel overview plot at high DPI: signal / anomaly score / indicator."""
    plt.style.use("seaborn-v0_8-paper")
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.titlesize": 13,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 8,
    })

    fig, axes = plt.subplots(3, 1, figsize=(11, 8), dpi=_PLOT_DPI)
    fig.patch.set_facecolor("white")

    n = len(timeseries)
    x_pct = np.linspace(0, 100, n)

    # Panel 1 — Input Signal
    ax1 = axes[0]
    ax1.plot(x_pct, timeseries, color="#2563EB", linewidth=1.3, alpha=0.9)
    ax1.set_title("Input Signal", fontweight="bold", pad=8)
    ax1.set_ylabel("Amplitude")
    ax1.grid(linestyle="--", alpha=0.45, color="#CBD5E1")
    ax1.set_xticks([0, 25, 50, 75, 100])
    ax1.set_xticklabels(["0", "25%", "50%", "75%", "100%"])
    ax1.set_facecolor("#F8FAFC")

    # Panel 2 — Anomaly Scores
    ax2 = axes[1]
    threshold = np.percentile(anomaly_scores, 90)
    high_mask = anomaly_scores > threshold
    ax2.plot(x_pct, anomaly_scores, color="#DC2626", linewidth=1.3, label="Score", zorder=3)
    ax2.fill_between(x_pct, 0, anomaly_scores, where=high_mask,
                     alpha=0.28, color="#DC2626", label="Anomaly Region", zorder=2)
    ax2.axhline(threshold, color="#374151", linestyle="--", linewidth=1.1,
                label=f"90th Percentile ({threshold:.3f})", zorder=4)
    ax2.set_title("Detected Anomaly Scores", fontweight="bold", pad=8)
    ax2.set_ylabel("Score")
    ax2.legend(loc="upper left", framealpha=0.85)
    ax2.grid(linestyle="--", alpha=0.45, color="#CBD5E1")
    ax2.set_xticks([0, 25, 50, 75, 100])
    ax2.set_xticklabels(["0", "25%", "50%", "75%", "100%"])
    ax2.set_facecolor("#F8FAFC")

    # Panel 3 — Binary Indicator
    ax3 = axes[2]
    indicator = high_mask.astype(int)
    ax3.fill_between(x_pct, 0, indicator, color="#DC2626", alpha=0.35, label="Anomaly Region")
    ax3.step(x_pct, indicator, color="#1E293B", linewidth=1.3, where="mid",
             label="Detected Anomalies")
    ax3.set_title("Anomaly Indicator", fontweight="bold", pad=8)
    ax3.set_ylabel("Indicator")
    ax3.set_xlabel("Signal Position (%)")
    ax3.set_ylim(-0.1, 1.4)
    ax3.set_yticks([0, 1])
    ax3.legend(loc="upper left", framealpha=0.85)
    ax3.grid(linestyle="--", alpha=0.45, color="#CBD5E1")
    ax3.set_xticks([0, 25, 50, 75, 100])
    ax3.set_xticklabels(["0", "25%", "50%", "75%", "100%"])
    ax3.set_facecolor("#F8FAFC")

    plt.tight_layout(pad=2.0)
    plt.savefig(save_path, dpi=_PLOT_DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _plot_tda_topology(tda_results: Dict[str, Any], save_path: str):
    """Generate TDA topology plot at high DPI: PCA point cloud + persistence diagram."""
    plt.style.use("seaborn-v0_8-paper")
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
    })

    fig = plt.figure(figsize=(13, 5.5), dpi=_PLOT_DPI)
    fig.patch.set_facecolor("white")

    # ── Left: PCA point cloud ─────────────────────────────────────────────────
    ax1 = fig.add_subplot(121, projection="3d")
    point_cloud = tda_results.get("subsampled_point_cloud", None)
    if point_cloud is not None and len(point_cloud) > 3:
        n_components = min(3, point_cloud.shape[1])
        pca = PCA(n_components=n_components)
        pca_coords = pca.fit_transform(point_cloud)
        if pca_coords.shape[1] < 3:
            pad = np.zeros((len(pca_coords), 3 - pca_coords.shape[1]))
            pca_coords = np.hstack([pca_coords, pad])
        colors = np.arange(len(pca_coords))
        sc = ax1.scatter(
            pca_coords[:, 0], pca_coords[:, 1], pca_coords[:, 2],
            c=colors, cmap="turbo", s=28, alpha=0.85,
            edgecolors="none",
        )
        ax1.set_title("Subsampled Point Cloud (PCA)", fontweight="bold", pad=10)
        ax1.set_xlabel("PC1", labelpad=4)
        ax1.set_ylabel("PC2", labelpad=4)
        ax1.set_zlabel("PC3", labelpad=4)
        ax1.view_init(elev=25, azim=45)
        ax1.xaxis.set_pane_color((0.97, 0.98, 1.0, 1.0))
        ax1.yaxis.set_pane_color((0.97, 0.98, 1.0, 1.0))
        ax1.zaxis.set_pane_color((0.97, 0.98, 1.0, 1.0))
        cbar = fig.colorbar(sc, ax=ax1, shrink=0.55, pad=0.08)
        cbar.set_label("Time Step", fontsize=8)
    else:
        ax1.text(0.5, 0.5, 0.5, "Point cloud\nnot available",
                 ha="center", va="center", fontsize=10)
        ax1.set_title("Subsampled Point Cloud (PCA)", fontweight="bold")

    # ── Right: Persistence diagram ────────────────────────────────────────────
    ax2 = fig.add_subplot(122)
    ax2.set_facecolor("#F8FAFC")
    barcode = tda_results.get("barcode", None)
    if barcode is not None and len(barcode) > 0:
        births = barcode[:, 0]
        deaths = barcode[:, 1]
        persistences = deaths - births
        finite_mask = np.isfinite(deaths)
        if not np.any(finite_mask):
            ax2.text(0.5, 0.5, "No finite persistence\npoints detected",
                     ha="center", va="center", transform=ax2.transAxes, fontsize=10)
        else:
            b = births[finite_mask]
            d = deaths[finite_mask]
            pers = persistences[finite_mask]
            sc2 = ax2.scatter(b, d, c=pers, cmap="plasma", s=55, alpha=0.88,
                              edgecolors="white", linewidth=0.4, zorder=3)
            lim_min = min(b.min(), d.min()) * 0.95
            lim_max = max(b.max(), d.max()) * 1.05
            ax2.plot([lim_min, lim_max], [lim_min, lim_max],
                     color="#64748B", linewidth=1.1, alpha=0.6, zorder=2)
            ax2.fill_between([lim_min, lim_max], [lim_min, lim_min],
                             [lim_min, lim_max], alpha=0.06, color="#94A3B8", zorder=1)
            cbar2 = fig.colorbar(sc2, ax=ax2, shrink=0.75)
            cbar2.set_label("Persistence", fontsize=8)
            ax2.set_xlim(lim_min, lim_max)
            ax2.set_ylim(lim_min, lim_max)
    else:
        ax2.text(0.5, 0.5, "Barcode data\nnot available",
                 ha="center", va="center", transform=ax2.transAxes, fontsize=10)

    ax2.set_title("Persistence Diagram", fontweight="bold", pad=10)
    ax2.set_xlabel("Birth Time")
    ax2.set_ylabel("Death Time")
    ax2.grid(linestyle="--", alpha=0.4, color="#CBD5E1")

    plt.tight_layout(pad=2.5)
    plt.savefig(save_path, dpi=_PLOT_DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
