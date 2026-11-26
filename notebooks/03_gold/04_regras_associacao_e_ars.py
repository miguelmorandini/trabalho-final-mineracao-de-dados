"""Camada Gold: regras de associacao para risco de evasao, analise de redes (ARS) e benchmark INEP."""

import sys
from pathlib import Path

import networkx as nx
import pandas as pd

# Garante acesso ao pacote utils na pasta notebooks/
notebooks_dir = Path(__file__).resolve().parent.parent
if (notebooks_dir / "utils").exists() and str(notebooks_dir) not in sys.path:
    sys.path.insert(0, str(notebooks_dir))

from utils.gold_utils import (
    get_gold_paths,
    save_gold_table,
)

ROOT_DIR, SILVER_DIR, GOLD_DIR = get_gold_paths()
print(f"Silver: {SILVER_DIR} | Gold: {GOLD_DIR}")

fato_360 = pd.read_parquet(GOLD_DIR / "fato_aluno_360.parquet")
fato_mat = pd.read_parquet(GOLD_DIR / "fato_matriculas.parquet")

# 1. Regras de associacao (cesta binaria de atributos de risco -> Evadido)
n_total = len(fato_360)
cesta = pd.DataFrame({
    "Evadido": fato_360["evadido_target"] == 1,
    "Nota_Abaixo_6": fato_360["nota_media"].fillna(5.0) < 6.0,
    "Sem_Acesso_AVA": fato_360["total_acessos"] == 0,
    "Inadimplente": fato_360["taxa_inadimplencia"].fillna(0.0) > 0.3,
    "Renda_Q1_Baixa": fato_360["faixa_renda"].astype(str) == "Q1 (Baixa)",
    "Multi_Matricula": fato_360["total_matriculas"] > 1,
})

antecedentes = ["Nota_Abaixo_6", "Sem_Acesso_AVA", "Inadimplente", "Renda_Q1_Baixa", "Multi_Matricula"]
sup_evadido = float(cesta["Evadido"].mean())

# Calcula suporte, confianca e lift para antecedentes unitarios
regras = []
for ant in antecedentes:
    sup_a = float(cesta[ant].mean())
    sup_ab = float((cesta[ant] & cesta["Evadido"]).mean())
    conf = sup_ab / sup_a if sup_a > 0 else 0.0
    lift = conf / sup_evadido if sup_evadido > 0 else 0.0
    regras.append({
        "antecedente": ant,
        "consequente": "Evadido",
        "suporte_pct": round(sup_ab * 100, 2),
        "confianca_pct": round(conf * 100, 2),
        "lift": round(lift, 4),
    })

# Calcula suporte, confianca e lift para pares de antecedentes (suporte minimo de 1%)
for i in range(len(antecedentes)):
    for j in range(i + 1, len(antecedentes)):
        a1, a2 = antecedentes[i], antecedentes[j]
        mask = cesta[a1] & cesta[a2]
        sup_a = float(mask.mean())
        if sup_a < 0.01:
            continue
        sup_ab = float((mask & cesta["Evadido"]).mean())
        conf = sup_ab / sup_a
        lift = conf / sup_evadido
        regras.append({
            "antecedente": f"{a1} + {a2}",
            "consequente": "Evadido",
            "suporte_pct": round(sup_ab * 100, 2),
            "confianca_pct": round(conf * 100, 2),
            "lift": round(lift, 4),
        })

regras_df = pd.DataFrame(regras).sort_values("lift", ascending=False)
save_gold_table(regras_df, "regras_associacao", GOLD_DIR)
print("=== Regras de Associação (Top Lift) ===")
print(regras_df.head(8).to_string(index=False))

# 2. Analise de Redes Sociais (ARS) no grafo bipartido Faculdade <-> Curso
edges = fato_mat.groupby(["faculdade", "curso"], as_index=False, observed=True).size()
G = nx.Graph()
for _, row in edges.iterrows():
    G.add_edge(f"FAC:{row['faculdade']}", f"CUR:{row['curso']}", weight=int(row["size"]))

deg_cent = nx.degree_centrality(G)
bet_cent = nx.betweenness_centrality(G)
ars_rows = [
    {
        "no": node,
        "tipo": "Faculdade" if node.startswith("FAC:") else "Curso",
        "grau": G.degree(node),
        "centralidade_grau": round(float(deg_cent[node]), 4),
        "intermediacao": round(float(bet_cent[node]), 4),
    }
    for node in G.nodes()
]
ars_df = pd.DataFrame(ars_rows).sort_values("centralidade_grau", ascending=False)
save_gold_table(ars_df, "ars_metricas_rede", GOLD_DIR)

# 3. Benchmark externo comparando com indicadores do Censo da Educacao Superior (INEP)
benchmark_df = pd.DataFrame([
    {"indicador": "Taxa de Evasão Global", "valor_nossa_base_pct": round(sup_evadido * 100, 2), "referencia_inep_pct": 28.0, "fonte": "INEP Censo Educação Superior (Ref. Nacional)"},
    {"indicador": "Taxa de Trancamento", "valor_nossa_base_pct": round(100 * float((fato_mat["status"] == "Trancado").mean()), 2), "referencia_inep_pct": 12.0, "fonte": "INEP Censo Educação Superior (Ref. Nacional)"},
])
save_gold_table(benchmark_df, "benchmark_inep", GOLD_DIR)
print("\n=== ARS & Benchmark salvos com sucesso ===")
print(f"Nós no grafo: {G.number_of_nodes()} | Arestas: {G.number_of_edges()} | Densidade: {round(nx.density(G), 4)}")
