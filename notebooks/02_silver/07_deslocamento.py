"""Camada Silver: padronizacao de tipos, limpeza textual, outliers e deduplicacao de deslocamento."""

import json
import sys
from pathlib import Path

# Garante acesso ao pacote utils na pasta notebooks/
notebooks_dir = Path(__file__).resolve().parent.parent
if (notebooks_dir / "utils").exists() and str(notebooks_dir) not in sys.path:
    sys.path.insert(0, str(notebooks_dir))

from utils.silver_utils import (
    detect_outliers_iqr,
    fix_mojibake_text,
    get_silver_paths,
    load_bronze,
    normalize_id_aluno,
    parse_br_float,
    parse_date_col,
    save_silver_outputs,
)

ROOT_DIR, BRONZE_DIR, SILVER_DIR = get_silver_paths()

df = load_bronze("deslocamento", BRONZE_DIR)
rows_in = len(df)

# Padroniza chaves e data de referencia
df["id_deslocamento"] = df["id_deslocamento"].astype(str).str.strip().astype("string")
df["id_aluno"] = normalize_id_aluno(df["id_aluno"])
df["data_referencia"] = parse_date_col(df["data_referencia"])

# Corrige mojibake e converte colunas descritivas para categoria
for col in ["cidade_origem", "regiao_origem", "meio_transporte"]:
    df[col], _ = fix_mojibake_text(df[col])
    df[col] = df[col].str.title().astype("category")

# Converte metricas numericas garantindo valores nao negativos e marca outliers IQR
for col in ["distancia_km", "tempo_minutos", "custo_diario"]:
    df[col] = parse_br_float(df[col]).clip(lower=0)

df["flag_dist_outlier_iqr"], iqr_dist = detect_outliers_iqr(df["distancia_km"])
df["flag_custo_outlier_iqr"], iqr_custo = detect_outliers_iqr(df["custo_diario"])

# Remove duplicatas pela chave id_deslocamento
dups_removed = int(df.duplicated(subset=["id_deslocamento"]).sum())
df = df.drop_duplicates(subset=["id_deslocamento"]).reset_index(drop=True)

# Registra log de transformacoes e salva tabela Silver em Parquet
log = {
    "dataset": "deslocamento",
    "rows_in": rows_in,
    "rows_out": int(len(df)),
    "duplicates_removed": dups_removed,
    "distancia_outliers_iqr": iqr_dist,
    "custo_outliers_iqr": iqr_custo,
}
save_silver_outputs(df, log, "deslocamento", SILVER_DIR)
print(json.dumps(log, ensure_ascii=False, indent=2))
print(df.head(5))
