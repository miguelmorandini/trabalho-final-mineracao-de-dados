"""Camada Silver: conversao de tipos, validacao de faixa [0, 10], deteccao de outliers e deduplicacao de notas."""

import json
import sys
from pathlib import Path

import numpy as np

# Garante acesso ao pacote utils na pasta notebooks/
notebooks_dir = Path(__file__).resolve().parent.parent
if (notebooks_dir / "utils").exists() and str(notebooks_dir) not in sys.path:
    sys.path.insert(0, str(notebooks_dir))

from utils.silver_utils import (
    detect_outliers_iqr,
    detect_outliers_zscore,
    fix_mojibake_text,
    get_silver_paths,
    load_bronze,
    normalize_id_aluno,
    parse_br_float,
    parse_date_col,
    save_silver_outputs,
)

ROOT_DIR, BRONZE_DIR, SILVER_DIR = get_silver_paths()

df = load_bronze("notas", BRONZE_DIR)
rows_in = len(df)

# Padroniza chave, texto da disciplina e data da avaliacao
df["id_aluno"] = normalize_id_aluno(df["id_aluno"])
df["disciplina"], mojibake_disc = fix_mojibake_text(df["disciplina"])
df["disciplina"] = df["disciplina"].str.title().astype("category")
df["data"] = parse_date_col(df["data"])

# Preserva nota_bruta, mantem NaN nas invalidas para analises sem distorcao e cria nota_clip [0, 10]
df["nota_bruta"] = parse_br_float(df["nota"])
invalida_mask = df["nota_bruta"].isna() | (df["nota_bruta"] < 0) | (df["nota_bruta"] > 10)
df["flag_nota_invalida"] = invalida_mask
df["nota"] = df["nota_bruta"].where(~invalida_mask, np.nan)
df["nota_clip"] = df["nota_bruta"].clip(lower=0.0, upper=10.0)

# Classifica situacao academica da nota
df["situacao_nota"] = np.where(
    df["flag_nota_invalida"],
    "Inválida (Fora 0-10)",
    np.where(df["nota"] >= 6.0, "Aprovado (>=6)", "Abaixo da Média (<6)"),
)
df["situacao_nota"] = df["situacao_nota"].astype("category")

# Identifica outliers por IQR e Z-Score nas notas validas
df["flag_nota_outlier_iqr"], iqr_info = detect_outliers_iqr(df["nota"])
df["flag_nota_outlier_zscore"], z_count = detect_outliers_zscore(df["nota"])

# Remove duplicatas exatas
dups_removed = int(df.duplicated().sum())
df = df.drop_duplicates().reset_index(drop=True)

# Registra log de transformacoes e salva tabela Silver em Parquet
log = {
    "dataset": "notas",
    "rows_in": rows_in,
    "rows_out": int(len(df)),
    "duplicates_removed": dups_removed,
    "mojibake_fixed_disciplina": mojibake_disc,
    "notas_invalidas_anuladas": int(invalida_mask.sum()),
    "nota_media_valida": round(float(df["nota"].mean()), 4),
    "nota_mediana_valida": round(float(df["nota"].median()), 4),
    "outliers_iqr": iqr_info,
    "outliers_zscore": z_count,
}
save_silver_outputs(df, log, "notas", SILVER_DIR)
print(json.dumps(log, ensure_ascii=False, indent=2))
print(df.head(5))
