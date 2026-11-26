"""Camada Silver: consolidacao dos logs de limpeza e validacao de integridade referencial."""

import json
import sys
from pathlib import Path

import pandas as pd

# Garante acesso ao pacote utils na pasta notebooks/
notebooks_dir = Path(__file__).resolve().parent.parent
if (notebooks_dir / "utils").exists() and str(notebooks_dir) not in sys.path:
    sys.path.insert(0, str(notebooks_dir))

from utils.silver_utils import get_silver_paths

_, _, SILVER_DIR = get_silver_paths()

# Carrega IDs validos de alunos na camada Silver para validar integridade referencial
alunos_ids = set(pd.read_parquet(SILVER_DIR / "alunos.parquet", columns=["id_aluno"])["id_aluno"].dropna())
tables = ["alunos", "matriculas", "notas", "acessos", "redes", "financeiro", "deslocamento", "frequencia"]

relatorio = {}
rows = []

# Agrega os logs individuais das 8 tabelas tratadas e calcula o match de id_aluno
for name in tables:
    with open(SILVER_DIR / f"{name}_silver_log.json", "r", encoding="utf-8") as f:
        log = json.load(f)
    ids = set(pd.read_parquet(SILVER_DIR / f"{name}.parquet", columns=["id_aluno"])["id_aluno"].dropna())
    match_pct = round(100.0 * len(ids & alunos_ids) / len(ids), 2) if ids else 0.0
    log["id_aluno_match_alunos_pct"] = match_pct
    relatorio[name] = log
    rows.append({
        "tabela": name,
        "linhas_bronze": log["rows_in"],
        "linhas_silver": log["rows_out"],
        "duplicatas_removidas": log["duplicates_removed"],
        "id_aluno_unicos": len(ids),
        "match_alunos_pct": match_pct,
    })

# Exporta resumo tabular em CSV e relatorio consolidado em JSON
resumo = pd.DataFrame(rows)
resumo.to_csv(SILVER_DIR / "resumo_silver.csv", index=False, encoding="utf-8-sig")
with open(SILVER_DIR / "relatorio_limpeza_silver.json", "w", encoding="utf-8") as f:
    json.dump(relatorio, f, ensure_ascii=False, indent=2)

print(resumo.to_string(index=False))
