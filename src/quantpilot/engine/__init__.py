"""Engine layer — factors, training, explanation, backtesting."""

from quantpilot.engine.factors import extract_factors, extract_factors_batch, get_factor_names
from quantpilot.engine.trainer import purged_walk_forward, save_model, load_model, model_exists
from quantpilot.engine.explainer import explain_global, explain_stock, format_shap_summary
from quantpilot.engine.backtester import backtest
from quantpilot.engine.registry import register_model, list_models, get_model_info

__all__ = [
    "extract_factors",
    "extract_factors_batch",
    "get_factor_names",
    "purged_walk_forward",
    "save_model",
    "load_model",
    "model_exists",
    "explain_global",
    "explain_stock",
    "format_shap_summary",
    "backtest",
    "register_model",
    "list_models",
    "get_model_info",
]
