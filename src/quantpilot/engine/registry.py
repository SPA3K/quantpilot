"""Model registry — tracks all available models (pretrained + custom)."""

import json
import logging
from datetime import datetime
from pathlib import Path

from quantpilot.config import REGISTRY_FILE, PREBUILT_DIR, CUSTOM_DIR

logger = logging.getLogger(__name__)


def _load_registry() -> dict:
    """Load registry from disk."""
    if REGISTRY_FILE.exists():
        return json.loads(REGISTRY_FILE.read_text())
    return {"pretrained": {}, "custom": {}}


def _save_registry(registry: dict):
    """Save registry to disk."""
    REGISTRY_FILE.parent.mkdir(parents=True, exist_ok=True)
    REGISTRY_FILE.write_text(json.dumps(registry, indent=2, ensure_ascii=False))


def register_model(
    sector_id: str,
    metrics: dict,
    is_custom: bool = False,
    user_id: str | None = None,
    metadata: dict | None = None,
):
    """Register a trained model in the registry."""
    registry = _load_registry()
    key = "custom" if is_custom else "pretrained"
    owner = user_id or "_system"

    if key not in registry:
        registry[key] = {}
    if owner not in registry[key]:
        registry[key][owner] = {}

    registry[key][owner][sector_id] = {
        "sector": sector_id,
        "registered_at": datetime.now().isoformat(),
        "metrics": metrics,
        "metadata": metadata or {},
    }

    _save_registry(registry)
    logger.info(f"Registered model: {key}/{owner}/{sector_id}")


def list_models(is_custom: bool = False, user_id: str | None = None) -> list[dict]:
    """List all models of a given type."""
    registry = _load_registry()
    key = "custom" if is_custom else "pretrained"
    owner = user_id or "_system"

    models = []
    owner_data = registry.get(key, {}).get(owner, {})
    for sector_id, info in owner_data.items():
        models.append({
            "sector": sector_id,
            "registered_at": info.get("registered_at"),
            "metrics": info.get("metrics", {}),
        })

    # Also check filesystem for models not in registry
    base = CUSTOM_DIR / user_id if is_custom and user_id else PREBUILT_DIR
    if base.exists():
        for d in base.iterdir():
            if d.is_dir() and (d / "model.lgb").exists():
                sid = d.name
                if not any(m["sector"] == sid for m in models):
                    # Model exists on disk but not in registry — add it
                    models.append({
                        "sector": sid,
                        "registered_at": None,
                        "metrics": {},
                        "source": "filesystem",
                    })

    return models


def get_model_info(sector_id: str, is_custom: bool = False, user_id: str | None = None) -> dict | None:
    """Get detailed info for a specific model."""
    registry = _load_registry()
    key = "custom" if is_custom else "pretrained"
    owner = user_id or "_system"

    info = registry.get(key, {}).get(owner, {}).get(sector_id)
    if info:
        return info

    # Fallback: try filesystem
    base = CUSTOM_DIR / user_id if is_custom and user_id else PREBUILT_DIR
    meta_path = base / sector_id / "metadata.json"
    if meta_path.exists():
        return json.loads(meta_path.read_text())

    return None
