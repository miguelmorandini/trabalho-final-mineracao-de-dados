"""Funções reutilizáveis de modelagem Star Schema, estatística e mineração para a Camada Gold."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


def get_gold_paths() -> tuple[Path, Path, Path]:
    """Resolve diretórios raiz, origem Silver e destino Gold."""
    cwd = Path.cwd().resolve()
    for candidate in [cwd, cwd.parent, cwd.parent.parent]:
        if (candidate / "data" / "silver").exists():
            silver_dir = candidate / "data" / "silver"
            gold_dir = candidate / "data" / "gold"
            gold_dir.mkdir(parents=True, exist_ok=True)
            return candidate, silver_dir, gold_dir
    raise FileNotFoundError(
        f"Não foi possível localizar 'data/silver' a partir de {cwd}"
    )


def save_gold_table(df: pd.DataFrame, name: str, gold_dir: Path, export_csv: bool = True) -> None:
    """Salva tabela Gold em Parquet e (opcionalmente) CSV UTF-8-SIG para importação direta no Power BI."""
    df.to_parquet(gold_dir / f"{name}.parquet", index=False, compression="snappy")
    if export_csv:
        df.to_csv(gold_dir / f"{name}.csv", index=False, encoding="utf-8-sig")


def save_gold_json(payload: dict, name: str, gold_dir: Path) -> Path:
    """Salva resumo analítico JSON na camada Gold."""
    out = gold_dir / f"{name}.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return out


def compute_descriptive_stats(df: pd.DataFrame, numeric_cols: list[str]) -> pd.DataFrame:
    """Calcula todas as métricas da tabela de Análise Descritiva (Aula 07/12)."""
    rows = []
    for col in numeric_cols:
        s = pd.to_numeric(df[col], errors="coerce").dropna()
        if s.empty:
            continue
        mean_val = float(s.mean())
        std_val = float(s.std())
        mode_series = s.mode()
        mode_val = float(mode_series.iloc[0]) if not mode_series.empty else np.nan
        cv = (std_val / mean_val * 100.0) if mean_val != 0 else np.nan

        rows.append({
            "variavel": col,
            "contagem": int(s.count()),
            "media": round(mean_val, 4),
            "mediana": round(float(s.median()), 4),
            "moda": round(mode_val, 4) if not np.isnan(mode_val) else np.nan,
            "minimo": round(float(s.min()), 4),
            "maximo": round(float(s.max()), 4),
            "amplitude": round(float(s.max() - s.min()), 4),
            "variancia": round(float(s.var()), 4),
            "desvio_padrao": round(std_val, 4),
            "coef_variacao_pct": round(cv, 2) if not np.isnan(cv) else np.nan,
            "p10_decil1": round(float(s.quantile(0.10)), 4),
            "q1_p25": round(float(s.quantile(0.25)), 4),
            "q2_p50": round(float(s.quantile(0.50)), 4),
            "q3_p75": round(float(s.quantile(0.75)), 4),
            "p90_decil9": round(float(s.quantile(0.90)), 4),
            "p95": round(float(s.quantile(0.95)), 4),
        })
    return pd.DataFrame(rows)
