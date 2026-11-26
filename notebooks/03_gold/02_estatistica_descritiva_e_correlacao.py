"""Camada Gold: estatistica descritiva, correlacoes (Pearson e Spearman) e distribuicao por faculdade."""

import sys
from pathlib import Path

import pandas as pd

# Garante acesso ao pacote utils na pasta notebooks/
notebooks_dir = Path(__file__).resolve().parent.parent
if (notebooks_dir / "utils").exists() and str(notebooks_dir) not in sys.path:
    sys.path.insert(0, str(notebooks_dir))

from utils.gold_utils import (
    compute_descriptive_stats,
    get_gold_paths,
    save_gold_table,
)

ROOT_DIR, SILVER_DIR, GOLD_DIR = get_gold_paths()
print(f"Silver: {SILVER_DIR} | Gold: {GOLD_DIR}")

fato_360 = pd.read_parquet(GOLD_DIR / "fato_aluno_360.parquet")

# 1. Estatisticas descritivas (tendencia central, dispersao e posicao relativa)
num_cols = [
    "idade", "renda_estimada", "nota_media", "tempo_logado_medio",
    "paginas_total", "interacoes_total", "valor_mensalidade_total", "taxa_inadimplencia",
]

desc_df = compute_descriptive_stats(fato_360, num_cols)
save_gold_table(desc_df, "estatisticas_descritivas", GOLD_DIR)
print("=== Estatísticas Descritivas ===")
print(desc_df[["variavel", "media", "mediana", "moda", "desvio_padrao", "coef_variacao_pct", "q1_p25", "q3_p75"]].to_string(index=False))

# 2. Correlacoes (Pearson e Spearman) e covariancia entre indicadores e evasao
corr_cols = [
    "evadido_target", "idade", "renda_estimada", "nota_media",
    "tempo_logado_medio", "paginas_total", "interacoes_total", "taxa_inadimplencia",
]
corr_sub = fato_360[corr_cols].dropna()

pearson = corr_sub.corr(method="pearson")
spearman = corr_sub.corr(method="spearman")
cov = corr_sub.cov()

corr_rows = []
for v1 in corr_cols:
    for v2 in corr_cols:
        if v1 < v2:
            corr_rows.append({
                "variavel_1": v1,
                "variavel_2": v2,
                "pearson_r": round(float(pearson.loc[v1, v2]), 4),
                "spearman_rho": round(float(spearman.loc[v1, v2]), 4),
                "covariancia": round(float(cov.loc[v1, v2]), 4),
            })

corr_df = pd.DataFrame(corr_rows).sort_values("pearson_r", key=lambda s: s.abs(), ascending=False)
save_gold_table(corr_df, "matriz_correlacao", GOLD_DIR)
print("\n=== Top Correlações (Pearson & Spearman) ===")
print(corr_df.head(10).to_string(index=False))

# 3. Distribuicao de frequencias e taxa de evasao por faculdade
freq_fac = (
    fato_360.groupby("faculdade_principal", as_index=False, observed=True)
    .agg(alunos=("id_aluno", "count"), evadidos=("evadido_target", "sum"), nota_media=("nota_media", "mean"))
)
freq_fac["taxa_evasao_pct"] = (freq_fac["evadidos"] / freq_fac["alunos"] * 100).round(2)
save_gold_table(freq_fac, "freq_faculdade", GOLD_DIR)
