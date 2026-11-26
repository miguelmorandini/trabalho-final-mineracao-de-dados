"""Camada Silver: conversao de tipos, validacao de nao negatividade, outliers e deduplicacao de redes."""

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
    save_silver_outputs,
)

ROOT_DIR, BRONZE_DIR, SILVER_DIR = get_silver_paths()

df = load_bronze("redes", BRONZE_DIR)
rows_in = len(df)

# Converte chave e contadores sociais para inteiros nao negativos
df["id_aluno"] = normalize_id_aluno(df["id_aluno"])
for col in ["interacoes", "seguidores", "seguindo"]:
    df[col] = pd.to_numeric(df[col], errors="coerce").clip(lower=0).astype("Int64")

# Detecta outliers nas interacoes por IQR e Z-Score
df["flag_interacoes_outlier_iqr"], iqr_int = detect_outliers_iqr(df["interacoes"].astype(float))
df["flag_interacoes_outlier_zscore"], z_int = detect_outliers_zscore(df["interacoes"].astype(float))

# Remove duplicatas exatas
dups_removed = int(df.duplicated().sum())
df = df.drop_duplicates().reset_index(drop=True)

# Registra log de transformacoes e salva tabela Silver em Parquet
log = {
    "dataset": "redes",
    "rows_in": rows_in,
    "rows_out": int(len(df)),
    "duplicates_removed": dups_removed,
    "interacoes_outliers_iqr": iqr_int,
    "interacoes_outliers_zscore": z_int,
}
save_silver_outputs(df, log, "redes", SILVER_DIR)
print(json.dumps(log, ensure_ascii=False, indent=2))
print(df.head(5))
