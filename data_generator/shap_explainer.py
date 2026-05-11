"""
analysis/shap_explainer.py
===========================
Anomaly detection model explainability via SHAP
(SHapley Additive exPlanations).

Usage:
    explainer = AnomalyShapExplainer.from_saved_model(
        model_path="ml/anomaly_model.pkl",
        scaler_path="ml/anomaly_scaler.pkl",
        features_path="ml/anomaly_features.json",
    )
    result = explainer.explain_instance(X_sample_row)
    print(result.top_features)

Dependencies: shap, scikit-learn, numpy, pandas, plotly
"""

import json
import pickle
import logging
from dataclasses import dataclass

import numpy as np
import plotly.graph_objects as go
import pandas as pd

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────
THRESHOLD_HIGH_RISK   = 0.65   # above this = confirmed anomaly
THRESHOLD_MEDIUM_RISK = 0.34   # above this = suspicious (production threshold)

# Display names for dashboard labels
FEATURE_DISPLAY_NAMES = {
    "cpu_utilization":     "CPU Utilization (%)",
    "memory_utilization":  "Memory Utilization (%)",
    "cpu_temp_c":          "CPU Temperature (°C)",
    "power_draw_w":        "Power Draw (W)",
    "network_rx_mbps":     "Network RX (Mbps)",
    "network_tx_mbps":     "Network TX (Mbps)",
    "disk_io_mbps":        "Disk I/O (Mbps)",
    "pue_contribution":    "PUE Contribution",
    "cpu_temp_roll_mean":  "CPU Temp — 10-sample Rolling Mean",
    "power_roll_std":      "Power Draw — 10-sample Rolling Std",
    "cpu_util_roll_mean":  "CPU Util — 10-sample Rolling Mean",
    "temp_zscore":         "CPU Temp — Z-Score",
}

# Semantic colors by risk direction
COLOR_INCREASE_RISK = "#FF4757"   # red    — feature increases anomaly risk
COLOR_DECREASE_RISK = "#00FF9F"   # green  — feature decreases anomaly risk
COLOR_NEUTRAL       = "#64748B"   # grey   — negligible impact


@dataclass
class ShapExplanation:
    """SHAP explanation result for a single instance."""
    server_id:      str
    timestamp:      str
    anomaly_score:  float
    is_anomaly:     bool
    risk_level:     str                 # 'high' | 'medium' | 'low'
    shap_values:    np.ndarray          # shape (n_features,)
    feature_names:  list
    feature_values: np.ndarray          # original (unscaled) values
    base_value:     float               # model expected value E[f(x)]
    top_features:   list                # top-5 features by |SHAP|, with direction
    diagnosis:      str                 # auto-generated diagnostic text
    recommendation: str                 # recommended action


@dataclass
class GlobalShapResult:
    """Global SHAP analysis result (entire test set)."""
    mean_abs_shap:   pd.Series          # mean absolute importance per feature
    shap_matrix:     np.ndarray         # shape (n_samples, n_features)
    feature_names:   list
    X_test_original: pd.DataFrame       # original feature values for scatter plots


class AnomalyShapExplainer:
    """
    SHAP explainability wrapper for the Random Forest anomaly detection model.

    TreeExplainer is optimized for tree-based models (Random Forest, XGBoost, etc.)
    and computes exact Shapley values in O(TLD²) where T=n_estimators, L=max_depth, D=n_features.
    """

    def __init__(self, model, scaler, feature_names: list):
        self.model         = model
        self.scaler        = scaler
        self.feature_names = feature_names
        self._shap_explainer = None

    @classmethod
    def from_saved_model(
        cls,
        model_path: str    = "ml/anomaly_model.pkl",
        scaler_path: str   = "ml/anomaly_scaler.pkl",
        features_path: str = "ml/anomaly_features.json",
    ) -> "AnomalyShapExplainer":
        with open(model_path,  "rb") as f:
            model = pickle.load(f)
        with open(scaler_path, "rb") as f:
            scaler = pickle.load(f)
        with open(features_path)     as f:
            feat = json.load(f)
        feature_names = feat if isinstance(feat, list) else feat.get("features", [])
        return cls(model, scaler, feature_names)

    def _get_explainer(self):
        """Lazy-load the TreeExplainer."""
        if self._shap_explainer is None:
            import shap
            self._shap_explainer = shap.TreeExplainer(
                self.model,
                feature_perturbation="interventional",
            )
        return self._shap_explainer

    def _risk_level(self, score: float) -> str:
        if score >= THRESHOLD_HIGH_RISK:
            return "high"
        if score >= THRESHOLD_MEDIUM_RISK:
            return "medium"
        return "low"

    def _auto_diagnosis(
        self,
        top_features: list,
        risk_level: str,
        feature_values: np.ndarray,
    ) -> tuple:
        """Generates a diagnostic text and action recommendation automatically."""
        if risk_level == "low":
            return "Server operating within normal parameters.", "No action required."

        # Identify the most likely failure pattern from top features
        feat_names_top = [f["name"] for f in top_features[:3] if f["direction"] == "increase"]
        diag_parts = []
        rec_parts  = []

        if any("temp" in n for n in feat_names_top):
            diag_parts.append("above-expected temperature drift")
            rec_parts.append("inspect rack cooling system")
        if any("power" in n for n in feat_names_top):
            diag_parts.append("power draw above baseline")
            rec_parts.append("check for high CPU/memory processes on the server")
        if any("cpu_util" in n for n in feat_names_top):
            diag_parts.append("anomalous CPU utilization pattern")
            rec_parts.append("scan for zombie or runaway processes")
        if any("zscore" in n for n in feat_names_top):
            diag_parts.append("statistically significant temperature deviation")
            rec_parts.append("monitor trend over the next 2 hours")

        prefix = "⚠️ Suspected anomaly" if risk_level == "medium" else "🔴 Confirmed anomaly"
        diag = f"{prefix}: {', '.join(diag_parts) or 'unclassified pattern'}."
        rec  = " | ".join(rec_parts) or "Escalate to infrastructure team."
        return diag, rec

    def explain_instance(
        self,
        X_row: pd.DataFrame,
        server_id: str = "srv_unknown",
        timestamp: str = "",
    ) -> ShapExplanation:
        """
        Explains a single DataFrame row (one server snapshot).

        Args:
            X_row:     DataFrame with 1 row and columns = feature_names
            server_id: Server ID for the report
            timestamp: Measurement timestamp

        Returns:
            ShapExplanation with SHAP values, top features and auto diagnosis
        """
        import shap  # noqa: F401

        # Keep original values before scaling
        original_values = X_row[self.feature_names].values.flatten()

        # Scale for inference
        X_scaled = self.scaler.transform(X_row[self.feature_names])

        # Anomaly score
        score = float(self.model.predict_proba(X_scaled)[0, 1])

        # SHAP values
        explainer = self._get_explainer()
        shap_vals = explainer.shap_values(X_scaled)
        # For binary RF, shap_values returns [class_0, class_1]
        if isinstance(shap_vals, list):
            shap_vals = shap_vals[1]
        sv = shap_vals[0]  # shape (n_features,)

        base_value = float(explainer.expected_value[1]
                           if isinstance(explainer.expected_value, (list, np.ndarray))
                           else explainer.expected_value)

        # Top-5 features by |SHAP|
        sorted_idx   = np.argsort(np.abs(sv))[::-1]
        top_features = []
        for i in sorted_idx[:5]:
            fn   = self.feature_names[i]
            sv_i = float(sv[i])
            top_features.append({
                "name":       fn,
                "display":    FEATURE_DISPLAY_NAMES.get(fn, fn),
                "shap_value": sv_i,
                "direction":  "increase" if sv_i > 0 else "decrease",
                "raw_value":  float(original_values[i]),
                "color":      COLOR_INCREASE_RISK if sv_i > 0 else COLOR_DECREASE_RISK,
            })

        risk      = self._risk_level(score)
        diag, rec = self._auto_diagnosis(top_features, risk, original_values)

        return ShapExplanation(
            server_id=server_id,
            timestamp=timestamp or pd.Timestamp.now().isoformat(),
            anomaly_score=score,
            is_anomaly=(score >= THRESHOLD_MEDIUM_RISK),
            risk_level=risk,
            shap_values=sv,
            feature_names=self.feature_names,
            feature_values=original_values,
            base_value=base_value,
            top_features=top_features,
            diagnosis=diag,
            recommendation=rec,
        )

    def explain_batch(
        self,
        X: pd.DataFrame,
        max_samples: int = 500,
    ) -> GlobalShapResult:
        """
        Computes SHAP values for a batch of samples (global analysis).
        Used for summary plots and global feature importance.
        """
        import shap  # noqa: F401

        sample = X[self.feature_names]
        if len(sample) > max_samples:
            sample = sample.sample(max_samples, random_state=42)

        X_scaled  = self.scaler.transform(sample)
        explainer = self._get_explainer()
        shap_vals = explainer.shap_values(X_scaled)
        if isinstance(shap_vals, list):
            shap_vals = shap_vals[1]

        mean_abs = pd.Series(
            np.abs(shap_vals).mean(axis=0),
            index=self.feature_names,
        ).sort_values(ascending=False)

        return GlobalShapResult(
            mean_abs_shap=mean_abs,
            shap_matrix=shap_vals,
            feature_names=self.feature_names,
            X_test_original=sample.reset_index(drop=True),
        )


# ── Plotly visualization helpers (for the dashboard) ─────────────────────

def make_waterfall_figure(explanation: ShapExplanation) -> "go.Figure":
    """
    Generates a SHAP Waterfall chart for one instance.
    Shows how each feature pushes the anomaly score up or down from the base value.
    """
    import plotly.graph_objects as go

    feats  = [f["display"] for f in explanation.top_features]
    values = [f["shap_value"] for f in explanation.top_features]

    all_feats  = ["E[f(x)] base"] + feats + ["Final Score"]
    all_values = [explanation.base_value] + values + [explanation.anomaly_score]

    fig = go.Figure(go.Waterfall(
        name="SHAP",
        orientation="v",
        measure=["absolute"] + ["relative"] * len(feats) + ["total"],
        x=all_feats,
        y=all_values,
        connector={"line": {"color": "rgba(255,255,255,0.3)", "width": 1}},
        increasing={"marker": {"color": COLOR_INCREASE_RISK}},
        decreasing={"marker": {"color": COLOR_DECREASE_RISK}},
        totals={"marker": {"color": "#00D4FF"}},
        textposition="outside",
        text=[f"{v:.3f}" for v in all_values],
    ))

    fig.add_hline(
        y=THRESHOLD_MEDIUM_RISK, line_dash="dash",
        line_color="rgba(255,176,32,0.7)",
        annotation_text="Threshold (0.34)",
        annotation_font_size=10,
    )

    fig.update_layout(
        title=f"SHAP Explanation — {explanation.server_id}",
        height=380,
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="JetBrains Mono, monospace", color="#F1F5F9", size=11),
        margin=dict(l=10, r=10, t=40, b=40),
        showlegend=False,
    )
    return fig


def make_bar_importance_figure(global_result: GlobalShapResult) -> "go.Figure":
    """Generates a global feature importance bar chart (mean |SHAP|)."""
    import plotly.graph_objects as go

    series  = global_result.mean_abs_shap.head(12)
    display = [FEATURE_DISPLAY_NAMES.get(n, n) for n in series.index]

    fig = go.Figure(go.Bar(
        x=series.values[::-1],
        y=display[::-1],
        orientation="h",
        marker=dict(
            color=series.values[::-1],
            colorscale=[[0, "#1A3A5C"], [0.5, "#2E75B6"], [1, "#00D4FF"]],
            line=dict(width=0),
        ),
        text=[f"{v:.4f}" for v in series.values[::-1]],
        textposition="outside",
    ))
    fig.update_layout(
        title="Global Feature Importance — Mean |SHAP Value|",
        height=420,
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="JetBrains Mono, monospace", color="#F1F5F9", size=11),
        margin=dict(l=10, r=10, t=40, b=10),
        xaxis_title="Mean |SHAP value|",
    )
    return fig


def make_beeswarm_figure(global_result: GlobalShapResult) -> "go.Figure":
    """
    Plotly version of the SHAP Beeswarm plot.
    Shows SHAP value distribution vs feature value for each top feature.
    """
    import plotly.graph_objects as go

    fig      = go.Figure()
    top_feats = global_result.mean_abs_shap.head(8).index.tolist()

    for i, feat in enumerate(top_feats[::-1]):
        fi      = global_result.feature_names.index(feat)
        sv_col  = global_result.shap_matrix[:, fi]
        raw_col = global_result.X_test_original[feat].values

        # Color by normalized feature value (0=blue, 1=red)
        raw_norm = (raw_col - raw_col.min()) / (raw_col.ptp() + 1e-9)

        fig.add_trace(go.Scatter(
            x=sv_col,
            y=[FEATURE_DISPLAY_NAMES.get(feat, feat)] * len(sv_col)
              + np.random.default_rng(i).normal(0, 0.08, len(sv_col)),
            mode="markers",
            name=feat,
            marker=dict(
                size=4,
                opacity=0.55,
                color=raw_norm,
                colorscale=[[0, "#1A6FAF"], [0.5, "#FFFFFF"], [1, "#FF4757"]],
                showscale=(i == 0),
                colorbar=dict(title="Feature<br>Value (norm.)", thickness=10, len=0.5)
                          if i == 0 else None,
            ),
            showlegend=False,
        ))

    fig.add_vline(x=0, line_color="rgba(255,255,255,0.3)", line_dash="dot")
    fig.update_layout(
        title="SHAP Beeswarm — Impact vs Feature Value",
        height=420,
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="JetBrains Mono, monospace", color="#F1F5F9", size=11),
        margin=dict(l=10, r=10, t=40, b=10),
        xaxis_title="SHAP value (impact on anomaly score)",
    )
    return fig
