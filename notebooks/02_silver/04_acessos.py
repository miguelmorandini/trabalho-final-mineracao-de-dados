"""Camada Silver: conversao de tipos, validacao de nao negatividade, outliers e deduplicacao de acessos."""

import json
import sys
from pathlib import Path

import pandas as pd

# Garante acesso ao pacote utils na pasta notebooks/
notebooks_dir = Path(__file__).resolve().parent.parent
if (notebooks_dir / "utils").exists() and str(notebooks_dir) not in sys.path:
    sys.path.insert(0, str(notebooks_dir))

from utils.silver_utils import (
    detect_outliers_iqr,
    detect_outliers_zscore,
    get_silver_paths,
    load_bronze,
    normalize_id_aluno,
    parse_br_float,
    parse_date_col,
    save_silver_outputs,
)

ROOT_DIR, BRONZE_DIR, SILVER_DIR = get_silver_paths()

df = load_bronze("acessos", BRONZE_DIR)
rows_in = len(df)

# Padroniza chave, data e metricas numericas garantindo valores nao negativos
df["id_aluno"] = normalize_id_aluno(df["id_aluno"])
df["data"] = parse_date_col(df["data"])
df["tempo_logado"] = parse_br_float(df["tempo_logado"]).clip(lower=0)
df["paginas"] = pd.to_numeric(df["paginas"], errors="coerce").clip(lower=0).astype("Int64")

# Detecta outliers por IQR e Z-Score
df["flag_tempo_outlier_iqr"], iqr_tempo = detect_outliers_iqr(df["tempo_logado"])
df["flag_tempo_outlier_zscore"], z_tempo = detect_outliers_zscore(df["tempo_logado"])
df["flag_paginas_outlier_iqr"], iqr_pag = detect_outliers_iqr(df["paginas"].astype(float))

# Remove duplicatas exatas
dups_removed = int(df.duplicated().sum())
df = df.drop_duplicates().reset_index(drop=True)

# Registra log de transformacoes e salva tabela Silver em Parquet
log = {
    "dataset": "acessos",
    "rows_in": rows_in,
    "rows_out": int(len(df)),
    "duplicates_removed": dups_removed,
    "tempo_outliers_iqr": iqr_tempo,
    "tempo_outliers_zscore": z_tempo,
    "paginas_outliers_iqr": iqr_pag,
}
save_silver_outputs(df, log, "acessos", SILVER_DIR)
print(json.dumps(log, ensure_ascii=False, indent=2))
print(df.head(5))
