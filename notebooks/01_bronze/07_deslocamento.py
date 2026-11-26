"""Camada Bronze: ingestao bruta e diagnostico inicial da tabela deslocamento."""

import json
import sys
from pathlib import Path

# Garante acesso ao pacote utils na pasta notebooks/
notebooks_dir = Path(__file__).resolve().parent.parent
if (notebooks_dir / "utils").exists() and str(notebooks_dir) not in sys.path:
    sys.path.insert(0, str(notebooks_dir))

from utils.bronze_utils import (
    get_project_paths,
    ingest_raw_file,
    profile_bronze_table,
    save_bronze_outputs,
)

# Resolve diretorios de entrada bruta e saida Bronze
ROOT_DIR, RAW_DIR, BRONZE_DIR = get_project_paths()
print(f"Origem: {RAW_DIR / 'Atividade11 - Deslocalamento aluno.csv'}")
print(f"Destino Bronze: {BRONZE_DIR}")

# Ingere o CSV bruto como string e gera o perfil de qualidade
df, meta = ingest_raw_file(
    dataset_name="deslocamento",
    filename="Atividade11 - Deslocalamento aluno.csv",
    sep=",",
    mixed_sep=False,
    raw_dir=RAW_DIR,
)

profile = profile_bronze_table(df, meta)
parquet_path, profile_path = save_bronze_outputs(df, profile, "deslocamento", BRONZE_DIR)

print(f"Linhas ingeridas: {meta['rows']:,} | Tamanho bruto: {meta['size_mb']} MB")
print(f"IDs únicos de aluno: {profile['unique_id_aluno']:,} | Duplicatas exatas: {profile['exact_duplicates']:,}")
print(f"Salvo em: {parquet_path.name} e {profile_path.name}")
print(df.head(5))

# Exibe campos vazios e anomalias detectadas para tratamento na Silver
print("=== Campos vazios por coluna ===")
print(json.dumps(profile["empty_counts"], ensure_ascii=False, indent=2))
if profile["anomalies"]:
    print("\n=== Anomalias detectadas para tratamento na Silver ===")
    print(json.dumps(profile["anomalies"], ensure_ascii=False, indent=2))
