"""Camada Bronze: consolidacao dos perfis e validacao de integridade referencial de id_aluno."""

import json
import sys
from pathlib import Path

import pandas as pd

# Garante acesso ao pacote utils na pasta notebooks/
notebooks_dir = Path(__file__).resolve().parent.parent
if (notebooks_dir / "utils").exists() and str(notebooks_dir) not in sys.path:
    sys.path.insert(0, str(notebooks_dir))

from utils.bronze_utils import get_project_paths

_, _, BRONZE_DIR = get_project_paths()

# Carrega conjunto de IDs validos da tabela de alunos para checar cobertura cruzada
alunos_ids = set(
    pd.read_parquet(BRONZE_DIR / "alunos.parquet", columns=["id_aluno"])["id_aluno"]
    .str.strip()
    .str.lstrip("0")
    .replace("", "0")
)

tables = ["alunos", "matriculas", "notas", "acessos", "redes", "financeiro", "deslocamento", "frequencia"]
consolidated = {}
rows = []

# Consolida metricas de cada perfil Bronze e calcula percentual de match com alunos
for name in tables:
    profile_path = BRONZE_DIR / f"{name}_profile.json"
    with open(profile_path, "r", encoding="utf-8") as f:
        prof = json.load(f)

    df_ids = pd.read_parquet(BRONZE_DIR / f"{name}.parquet", columns=["id_aluno"])["id_aluno"].str.strip()
    non_empty = set(df_ids[df_ids != ""].str.lstrip("0").replace("", "0"))
    match_pct = round(100.0 * len(non_empty & alunos_ids) / len(non_empty), 2) if non_empty else 0.0
    prof["id_aluno_match_alunos_pct"] = match_pct
    consolidated[name] = prof

    rows.append({
        "tabela": name,
        "linhas": prof["metadata"]["rows"],
        "colunas": len(prof["metadata"]["columns"]),
        "id_aluno_unicos": prof["unique_id_aluno"],
        "match_alunos_pct": match_pct,
        "duplicatas_exatas": prof["exact_duplicates"],
        "total_campos_vazios": prof["total_empty_fields"],
    })

# Exporta tabela resumo em CSV e relatorio completo em JSON
resumo_df = pd.DataFrame(rows)
resumo_df.to_csv(BRONZE_DIR / "resumo_bronze.csv", index=False, encoding="utf-8-sig")
with open(BRONZE_DIR / "relatorio_qualidade_bronze.json", "w", encoding="utf-8") as f:
    json.dump(consolidated, f, ensure_ascii=False, indent=2)

print(resumo_df.to_string(index=False))
