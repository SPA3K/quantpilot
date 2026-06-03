"""SHAP-based model explanation — global and per-stock factor contributions."""

import json
import logging

import numpy as np
import shap
import lightgbm as lgb

logger = __import__("logging").getLogger(__name__)


def explain_global(model: lgb.LGBMRanker, X: np.ndarray, factor_cols: list[str]) -> dict:
    """Compute global SHAP feature importance.

    Returns dict with top factors and their importance scores.
    """
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    # Mean absolute SHAP value per feature
    if isinstance(shap_values, list):
        shap_values = shap_values[0]

    importance = np.mean(np.abs(shap_values), axis=0)
    factor_importance = sorted(
        zip(factor_cols, importance),
        key=lambda x: x[1],
        reverse=True,
    )

    return {
        "top_factors": [
            {
                "name": name,
                "importance": round(float(imp), 4),
                "rank": i + 1,
            }
            for i, (name, imp) in enumerate(factor_importance[:15])
        ],
        "total_features": len(factor_cols),
    }


def explain_stock(
    model: lgb.LGBMRanker,
    x_row: np.ndarray,
    factor_cols: list[str],
    feature_names_map: dict | None = None,
) -> list[dict]:
    """Explain a single stock's prediction.

    Returns list of factor contributions sorted by absolute value.
    """
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(x_row.reshape(1, -1))

    if isinstance(shap_values, list):
        shap_values = shap_values[0]

    values = shap_values[0] if shap_values.ndim > 1 else shap_values

    contributions = []
    for i, (name, val) in enumerate(zip(factor_cols, values)):
        contributions.append({
            "factor": name,
            "value": round(float(x_row[i]), 4),
            "contribution": round(float(val), 4),
        })

    # Sort by absolute contribution
    contributions.sort(key=lambda x: abs(x["contribution"]), reverse=True)
    return contributions[:10]  # Top 10


def format_shap_summary(contributions: list[dict]) -> str:
    """Format SHAP contributions into human-readable text.

    Example output:
        📈 revenue_growth_yoy: +35% (+0.18)
        ⚠️ volatility_20d: 3.2% (-0.08)
    """
    lines = []
    for c in contributions:
        name = c["factor"]
        val = c["value"]
        contrib = c["contribution"]

        if contrib > 0:
            emoji = "📈"
            sign = "+"
        else:
            emoji = "⚠️"
            sign = ""

        lines.append(f"{emoji} {name}: {val} ({sign}{contrib:.3f})")

    return "\n".join(lines)
