"""Train/val/test splits.

random   : stratified random split (setup B) - optimistic, pools from the same
           week/creator/scam wave end up on both sides.
temporal : split by t0 (pool launch). With purge=True, training rows whose label
           horizon [t0, t0+N+H] reaches into the next period are dropped, so no
           training label depends on events that happen after validation/test starts.
"""
from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

DAY = 86400


def _ts(d: str) -> int:
    return int(datetime.fromisoformat(d).replace(tzinfo=timezone.utc).timestamp())


def random_split(df: pd.DataFrame, seed: int = 42, val_size: float = 0.15, test_size: float = 0.15):
    idx = np.arange(len(df))
    tr, te = train_test_split(idx, test_size=test_size, random_state=seed, stratify=df.y)
    tr, va = train_test_split(tr, test_size=val_size / (1 - test_size), random_state=seed, stratify=df.y.iloc[tr])
    return df.iloc[tr], df.iloc[va], df.iloc[te]


def temporal_split(df: pd.DataFrame, train_end: str, val_end: str, test_end: str | None = None,
                   h_days: float = 30, purge: bool = True, drop_seen_creators: bool = False):
    a, b = _ts(train_end), _ts(val_end)
    c = _ts(test_end) if test_end else np.inf
    t0 = df.t0.to_numpy()
    horizon_end = df.t_end.to_numpy() + h_days * DAY
    tr = t0 < a
    va = (t0 >= a) & (t0 < b)
    te = (t0 >= b) & (t0 < c)
    if purge:
        tr &= horizon_end <= a
        va &= horizon_end <= b
    train, val, test = df[tr], df[va], df[te]
    if drop_seen_creators:  # "cold-start" evaluation: creators never seen in training
        seen = set(train.creator.dropna()) if "creator" in train else set(train.token_creator.dropna())
        col = "creator" if "creator" in test else "token_creator"
        test = test[~test[col].isin(seen)]
    return train, val, test


def make_split(df: pd.DataFrame, spec: dict, h_days: float, seed: int):
    kind = spec.get("kind", "temporal")
    if kind == "random":
        return random_split(df, seed=seed, val_size=spec.get("val_size", 0.15), test_size=spec.get("test_size", 0.15))
    if kind == "temporal":
        train, val, test = temporal_split(
            df, spec["train_end"], spec["val_end"], spec.get("test_end"), h_days=h_days,
            purge=spec.get("purge", True), drop_seen_creators=spec.get("drop_seen_creators", False))
        if spec.get("drop_seen_templates"):
            if "bytecode_hash" not in df:
                raise ValueError("drop_seen_templates requires bytecode_hash in the processed dataset")
            seen = set(train.bytecode_hash.dropna())
            test = test[test.bytecode_hash.isna() | ~test.bytecode_hash.isin(seen)]
        return train, val, test
    raise ValueError(kind)


def make_splits(df: pd.DataFrame, spec: dict, h_days: float, seed: int):
    """Return named splits; rolling_temporal expands to one result per configured fold."""
    if spec.get("kind") != "rolling_temporal":
        return [("main", *make_split(df, spec, h_days, seed))]
    common = {k: v for k, v in spec.items() if k not in ("kind", "folds")}
    out = []
    for i, fold in enumerate(spec.get("folds", []), 1):
        fspec = {**common, **fold, "kind": "temporal"}
        out.append((fold.get("name", f"fold{i}"), *make_split(df, fspec, h_days, seed)))
    if not out:
        raise ValueError("rolling_temporal requires at least one split.folds entry")
    return out
