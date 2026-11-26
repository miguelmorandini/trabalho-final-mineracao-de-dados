"""Camada Silver: padronizacao de tipos, limpeza textual e deduplicacao da micro-amostra de frequencia."""

import json
import sys
from pathlib import Path

import pandas as pd

# Garante acesso ao pacote utils na pasta notebooks/
notebooks_dir = Path(__file__).resolve().parent.parent
if (notebooks_dir / "utils").exists() and str(notebooks_dir) not in sys.path:
    sys.path.insert(0, str(notebooks_dir))

from utils.silver_utils import (
    fix_mojibake_text,
    get_silver_paths,
    load_bronze,
    normalize_id_aluno,
    parse_date_col,
    save_silver_outputs,
)

ROOT_DIR, BRONZE_DIR, SILVER_DIR = get_silver_paths()

df = load_bronze("frequencia", BRONZE_DIR)
rows_in = len(df)

# Padroniza chaves, data da aula e minutos de atraso (>= 0)
df["id_frequencia"] = df["id_frequencia"].astype(str).str.strip().astype("string")
df["id_aluno"] = normalize_id_aluno(df["id_aluno"])
df["data_aula"] = parse_date_col(df["data_aula"])
df["minutos_atraso"] = pd.to_numeric(df["minutos_atraso"], errors="coerce").clip(lower=0).fillna(0).astype("Int64")

# Corrige mojibake e converte colunas categoricas
for col in ["curso", "disciplina", "turno", "situacao"]:
    df[col], _ = fix_mojibake_text(df[col])
    df[col] = df[col].str.title().astype("category")

# Remove duplicatas pela chave id_frequencia
dups_removed = int(df.duplicated(subset=["id_frequencia"]).sum())
df = df.drop_duplicates(subset=["id_frequencia"]).reset_index(drop=True)

# Registra log com observacao do escopo amostral e salva tabela Silver em Parquet
log = {
    "dataset": "frequencia",
    "rows_in": rows_in,
    "rows_out": int(len(df)),
    "duplicates_removed": dups_removed,
    "escopo": "Micro-amostra complementar (20 registros / 9 alunos).",
}
save_silver_outputs(df, log, "frequencia", SILVER_DIR)
print(json.dumps(log, ensure_ascii=False, indent=2))
print(df.head(5))
