"""Camada Silver: limpeza de chaves, auditoria de inconsistencias, imputacao, padronizacao monetaria e outliers de financeiro."""

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

df = load_bronze("financeiro", BRONZE_DIR)
rows_in = len(df)

# Normaliza chaves e descarta registros sem id_aluno por nao permitirem vinculo ao aluno
df["id_lancamento"] = normalize_id_aluno(df["id_lancamento"])
df["id_aluno"] = normalize_id_aluno(df["id_aluno"])
sem_id = int(df["id_aluno"].isna().sum())
df = df.dropna(subset=["id_aluno"]).copy()

# Padroniza competencia e data de pagamento
df["competencia"] = df["competencia"].astype(str).str.strip().replace("", "Não Informada")
df["data_pagamento"] = parse_date_col(df["data_pagamento"])

# Auditoria de inconsistencias financeiras antes da correcao
mens_raw = parse_br_float(df["valor_mensalidade"])
pago_raw = parse_br_float(df["valor_pago"])
sit_raw = df["situacao_financeira"].astype(str).str.strip().str.upper()
df["flag_mensalidade_negativa"] = (mens_raw < 0).fillna(False)
df["flag_pago_valor_zero"] = ((pago_raw == 0) & (sit_raw == "PAGO")).fillna(False)
inconsist_mens_neg = int(df["flag_mensalidade_negativa"].sum())
inconsist_pago_zero = int(df["flag_pago_valor_zero"].sum())

for col in ["valor_mensalidade", "valor_desconto", "valor_pago"]:
    df[col] = parse_br_float(df[col]).clip(lower=0)

# Imputa valor_mensalidade ausente pela mediana e preenche desconto/pago com 0.0
mens_na = int(df["valor_mensalidade"].isna().sum())
mens_mediana = float(df["valor_mensalidade"].median())
df["valor_mensalidade"] = df["valor_mensalidade"].fillna(mens_mediana)
df["valor_desconto"] = df["valor_desconto"].fillna(0.0)
df["valor_pago"] = df["valor_pago"].fillna(0.0)

# Guarda categoria original vazia como 'Sem situacao' antes da imputacao pela moda
df["situacao_financeira_raw"] = (
    sit_raw.replace("", "Sem situação").str.replace("_", " ").str.capitalize().astype("category")
)
df["situacao_financeira"] = sit_raw.replace("", pd.NA)
moda_sit = str(df["situacao_financeira"].mode(dropna=True).iloc[0])
sit_na = int(df["situacao_financeira"].isna().sum())
df["situacao_financeira"] = df["situacao_financeira"].fillna(moda_sit).astype("category")

# Trata forma_pagamento vazia como categoria explicita para titulos nao pagos
df["forma_pagamento"] = (
    df["forma_pagamento"].astype(str).str.strip().replace("", "NÃO PAGO / N/A").str.upper().astype("category")
)

# Detecta outliers em valor_mensalidade por IQR e Z-Score
df["flag_mensalidade_outlier_iqr"], iqr_mens = detect_outliers_iqr(df["valor_mensalidade"])
df["flag_mensalidade_outlier_zscore"], z_mens = detect_outliers_zscore(df["valor_mensalidade"])

# Remove duplicatas pela chave id_lancamento
dups_removed = int(df.duplicated(subset=["id_lancamento"]).sum())
df = df.drop_duplicates(subset=["id_lancamento"]).reset_index(drop=True)

# Registra log de transformacoes e salva tabela Silver em Parquet
log = {
    "dataset": "financeiro",
    "rows_in": rows_in,
    "rows_removed_null_id_aluno": sem_id,
    "rows_out": int(len(df)),
    "duplicates_removed": dups_removed,
    "valor_mensalidade_imputado_mediana": mens_na,
    "valor_mensalidade_mediana": round(mens_mediana, 2),
    "situacao_imputada_moda": sit_na,
    "situacao_moda": moda_sit,
    "inconsist_mensalidade_negativa": inconsist_mens_neg,
    "inconsist_pago_valor_zero": inconsist_pago_zero,
    "mensalidade_outliers_iqr": iqr_mens,
    "mensalidade_outliers_zscore": z_mens,
}
save_silver_outputs(df, log, "financeiro", SILVER_DIR)
print(json.dumps(log, ensure_ascii=False, indent=2))
print(df.head(5))
