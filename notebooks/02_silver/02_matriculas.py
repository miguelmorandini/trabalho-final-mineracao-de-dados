"""Camada Silver: padronizacao de tipos, limpeza textual e deduplicacao de matriculas."""

import json
import sys
from pathlib import Path

# Garante acesso ao pacote utils na pasta notebooks/
notebooks_dir = Path(__file__).resolve().parent.parent
if (notebooks_dir / "utils").exists() and str(notebooks_dir) not in sys.path:
    sys.path.insert(0, str(notebooks_dir))

from utils.silver_utils import (
    fix_mojibake_text,
    get_silver_paths,
    load_bronze,
    normalize_id_aluno,
    save_silver_outputs,
)

ROOT_DIR, BRONZE_DIR, SILVER_DIR = get_silver_paths()

df = load_bronze("matriculas", BRONZE_DIR)
rows_in = len(df)

# Converte chaves para Int64 anulavel
df["id_matricula"] = normalize_id_aluno(df["id_matricula"])
df["id_aluno"] = normalize_id_aluno(df["id_aluno"])

# Corrige mojibake e padroniza formato das colunas categoricas
for col in ["faculdade", "curso", "semestre", "status"]:
    df[col], _ = fix_mojibake_text(df[col])

df["faculdade"] = df["faculdade"].str.upper().astype("category")
df["curso"] = df["curso"].str.title().astype("category")
df["semestre"] = df["semestre"].str.strip().astype("category")
df["status"] = df["status"].str.title().astype("category")

# Remove duplicatas pela chave id_matricula
dups_removed = int(df.duplicated(subset=["id_matricula"]).sum())
df = df.drop_duplicates(subset=["id_matricula"]).reset_index(drop=True)

# Registra log de transformacoes e salva tabela Silver em Parquet
log = {
    "dataset": "matriculas",
    "rows_in": rows_in,
    "rows_out": int(len(df)),
    "duplicates_removed": dups_removed,
    "status_counts": {str(k): int(v) for k, v in df["status"].value_counts().items()},
}
save_silver_outputs(df, log, "matriculas", SILVER_DIR)
print(json.dumps(log, ensure_ascii=False, indent=2))
print(df.head(5))
