"""Funções reutilizáveis de limpeza, padronização, outliers e LGPD para a Camada Silver."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd


def get_silver_paths() -> tuple[Path, Path, Path]:
    """Resolve diretórios raiz, origem Bronze e destino Silver."""
    cwd = Path.cwd().resolve()
    for candidate in [cwd, cwd.parent, cwd.parent.parent]:
        if (candidate / "data" / "bronze").exists():
            bronze_dir = candidate / "data" / "bronze"
            silver_dir = candidate / "data" / "silver"
            silver_dir.mkdir(parents=True, exist_ok=True)
            return candidate, bronze_dir, silver_dir
    raise FileNotFoundError(
        f"Não foi possível localizar 'data/bronze' a partir de {cwd}"
    )


def load_bronze(dataset_name: str, bronze_dir: Path) -> pd.DataFrame:
    """Carrega tabela da camada Bronze removendo colunas técnicas de controle."""
    df = pd.read_parquet(bronze_dir / f"{dataset_name}.parquet")
    biz_cols = [c for c in df.columns if not c.startswith("_bronze_")]
    return df[biz_cols].copy()


def normalize_id_aluno(series: pd.Series) -> pd.Series:
    """Converte id_aluno string (com zeros à esquerda) para Int64 anulável."""
    cleaned = series.astype(str).str.strip().str.lstrip("0").replace({"": pd.NA, "nan": pd.NA, "None": pd.NA})
    # Caso o ID fosse literalmente '0000000', lstrip('0') vira '' — tratamos apenas strings vazias originais como NA
    orig_empty = series.astype(str).str.strip() == ""
    num = pd.to_numeric(cleaned, errors="coerce").astype("Int64")
    num[orig_empty] = pd.NA
    return num


def fix_mojibake_text(series: pd.Series) -> tuple[pd.Series, int]:
    """Corrige texto UTF-8 lido incorretamente como Latin-1 (ex.: 'MatÃ£o' -> 'Matão')."""
    s = series.astype(str).str.strip()
    mask = s.str.contains("Ã|Â", regex=True, na=False)
    fixed_count = int(mask.sum())
    if fixed_count > 0:
        def _repair(val: str) -> str:
            try:
                return val.encode("latin-1").decode("utf-8")
            except (UnicodeEncodeError, UnicodeDecodeError):
                return val
        s.loc[mask] = s.loc[mask].map(_repair)
    s = s.str.replace(r"\s+", " ", regex=True).str.strip()
    return s, fixed_count


def parse_br_float(series: pd.Series) -> pd.Series:
    """Converte strings numéricas com vírgula decimal para float64."""
    s = series.astype(str).str.strip().replace("", pd.NA)
    s = s.str.replace(",", ".", regex=False)
    return pd.to_numeric(s, errors="coerce").astype("float64")


def parse_date_col(series: pd.Series) -> pd.Series:
    """Converte coluna textual de data para datetime64[ns] tolerando múltiplos formatos."""
    s = series.astype(str).str.strip().replace({"": pd.NA, "None": pd.NA, "nan": pd.NA})
    return pd.to_datetime(s, errors="coerce", format="mixed")


def anonymize_name_lgpd(series: pd.Series, id_series: pd.Series) -> pd.Series:
    """Anonimiza nomes (LGPD) gerando pseudônimo determinístico via SHA-256 truncado."""
    def _hash_row(val: str, aid) -> str:
        raw = f"{aid}:{val}".encode("utf-8", errors="ignore")
        return "ANON_" + hashlib.sha256(raw).hexdigest()[:12]
    return pd.Series(
        [_hash_row(v, aid) for v, aid in zip(series.astype(str), id_series)],
        index=series.index,
        dtype="string",
    )


def detect_outliers_iqr(series: pd.Series, factor: float = 1.5) -> tuple[pd.Series, dict]:
    """Detecta outliers usando o método IQR (Q1 - 1.5*IQR, Q3 + 1.5*IQR)."""
    valid = series.dropna()
    if valid.empty:
        return pd.Series(False, index=series.index), {"q1": None, "q3": None, "iqr": None, "outliers": 0}
    q1 = float(valid.quantile(0.25))
    q3 = float(valid.quantile(0.75))
    iqr = q3 - q1
    low = q1 - factor * iqr
    high = q3 + factor * iqr
    mask = (series < low) | (series > high)
    mask = mask.fillna(False)
    return mask, {
        "q1": round(q1, 4),
        "q3": round(q3, 4),
        "iqr": round(iqr, 4),
        "lower_bound": round(low, 4),
        "upper_bound": round(high, 4),
        "outliers": int(mask.sum()),
    }


def detect_outliers_zscore(series: pd.Series, threshold: float = 3.0) -> tuple[pd.Series, int]:
    """Detecta outliers usando Z-Score (|z| > threshold)."""
    valid = series.dropna()
    std = float(valid.std()) if len(valid) > 1 else 0.0
    if std == 0.0:
        return pd.Series(False, index=series.index), 0
    mean = float(valid.mean())
    z = (series - mean) / std
    mask = (z.abs() > threshold).fillna(False)
    return mask, int(mask.sum())


def save_silver_outputs(
    df: pd.DataFrame, log_dict: dict, dataset_name: str, silver_dir: Path
) -> tuple[Path, Path]:
    """Salva tabela limpa na camada Silver (Parquet) e log auditável de transformações (JSON)."""
    parquet_path = silver_dir / f"{dataset_name}.parquet"
    log_path = silver_dir / f"{dataset_name}_silver_log.json"
    df.to_parquet(parquet_path, index=False, compression="snappy")
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(log_dict, f, ensure_ascii=False, indent=2)
    return parquet_path, log_path
