"""Camada Gold: construcao do Star Schema completo (Parquet/CSV) e visao fato_aluno_360."""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

notebooks_dir = Path(__file__).resolve().parent.parent
if (notebooks_dir / "utils").exists() and str(notebooks_dir) not in sys.path:
    sys.path.insert(0, str(notebooks_dir))

from utils.gold_utils import (
    get_gold_paths,
    save_gold_json,
    save_gold_table,
)

ROOT_DIR, SILVER_DIR, GOLD_DIR = get_gold_paths()
print(f"Silver: {SILVER_DIR} | Gold: {GOLD_DIR}")

alunos = pd.read_parquet(SILVER_DIR / "alunos.parquet")
matriculas = pd.read_parquet(SILVER_DIR / "matriculas.parquet")
notas = pd.read_parquet(SILVER_DIR / "notas.parquet")
acessos = pd.read_parquet(SILVER_DIR / "acessos.parquet")
redes = pd.read_parquet(SILVER_DIR / "redes.parquet")
financeiro = pd.read_parquet(SILVER_DIR / "financeiro.parquet")
deslocamento = pd.read_parquet(SILVER_DIR / "deslocamento.parquet")
frequencia = pd.read_parquet(SILVER_DIR / "frequencia.parquet")

# 1. Dimensao dim_curso
dim_curso = (
    matriculas[["faculdade", "curso"]]
    .drop_duplicates()
    .sort_values(["faculdade", "curso"])
    .reset_index(drop=True)
)
dim_curso["id_curso"] = np.arange(1, len(dim_curso) + 1, dtype="int64")
dim_curso = dim_curso[["id_curso", "faculdade", "curso"]]
save_gold_table(dim_curso, "dim_curso", GOLD_DIR)

# 2. Dimensao dim_calendario cobrindo o intervalo de notas e pagamentos
datas_validas = pd.concat([notas["data"], financeiro["data_pagamento"]]).dropna()
dt_min = datas_validas.min().normalize()
dt_max = datas_validas.max().normalize()
cal_range = pd.date_range(dt_min, dt_max, freq="D")
dim_calendario = pd.DataFrame({"data": cal_range})
dim_calendario["ano"] = dim_calendario["data"].dt.year
dim_calendario["mes"] = dim_calendario["data"].dt.month
dim_calendario["ano_mes"] = dim_calendario["data"].dt.strftime("%Y-%m")
dim_calendario["trimestre"] = "T" + dim_calendario["data"].dt.quarter.astype(str)
save_gold_table(dim_calendario, "dim_calendario", GOLD_DIR)

# 3. Dimensao dim_aluno (LGPD + flags de cobertura, qualidade, faixas e engajamento para Star Schema no BI)
mat_ids = set(matriculas["id_aluno"].dropna())
nota_ids = set(notas["id_aluno"].dropna())
acesso_ids = set(acessos["id_aluno"].dropna())
redes_ids = set(redes["id_aluno"].dropna())
fin_ids = set(financeiro["id_aluno"].dropna())
desl_ids = set(deslocamento["id_aluno"].dropna())
freq_ids = set(frequencia["id_aluno"].dropna())

dim_aluno = alunos[[
    "id_aluno", "nome_anonimizado", "sexo", "idade", "faixa_etaria", "faixa_etaria_detalhada",
    "cidade", "renda_estimada", "faixa_renda", "faixa_renda_detalhada",
    "flag_sexo_ausente", "flag_cidade_ausente", "flag_idade_ausente", "flag_renda_ausente",
]].copy()

dim_aluno["tem_matricula"] = dim_aluno["id_aluno"].isin(mat_ids)
dim_aluno["tem_nota"] = dim_aluno["id_aluno"].isin(nota_ids)
dim_aluno["tem_acesso"] = dim_aluno["id_aluno"].isin(acesso_ids)
dim_aluno["tem_redes"] = dim_aluno["id_aluno"].isin(redes_ids)
dim_aluno["tem_financeiro"] = dim_aluno["id_aluno"].isin(fin_ids)
dim_aluno["tem_deslocamento"] = dim_aluno["id_aluno"].isin(desl_ids)
dim_aluno["tem_frequencia"] = dim_aluno["id_aluno"].isin(freq_ids)
dim_aluno["tem_complementar"] = (
    dim_aluno["tem_matricula"]
    | dim_aluno["tem_nota"]
    | dim_aluno["tem_acesso"]
    | dim_aluno["tem_redes"]
    | dim_aluno["tem_financeiro"]
    | dim_aluno["tem_deslocamento"]
    | dim_aluno["tem_frequencia"]
)
dim_aluno["qtd_fontes"] = (
    dim_aluno["tem_matricula"].astype("int8")
    + dim_aluno["tem_nota"].astype("int8")
    + dim_aluno["tem_acesso"].astype("int8")
    + dim_aluno["tem_redes"].astype("int8")
    + dim_aluno["tem_financeiro"].astype("int8")
    + dim_aluno["tem_deslocamento"].astype("int8")
    + dim_aluno["tem_frequencia"].astype("int8")
)

# Ordenacoes explicita (1-para-1) para sortByColumn no Power BI
dim_aluno["faixa_etaria_detalhada"] = dim_aluno["faixa_etaria_detalhada"].astype(str).replace({"nan": None, "<NA>": None})
etaria_order = {"Até 17": 1, "18–29": 2, "30–39": 3, "40–49": 4, "50–59": 5, "60+": 6}
dim_aluno["ordem_faixa_etaria"] = dim_aluno["faixa_etaria_detalhada"].map(etaria_order).fillna(99).astype("int8")

dim_aluno["faixa_renda_detalhada"] = dim_aluno["faixa_renda_detalhada"].astype(str).replace({"nan": None, "<NA>": None})
renda_order = {"Até R$ 2 mil": 1, "R$ 2–4 mil": 2, "R$ 4–6 mil": 3, "R$ 6–8 mil": 4, "Acima de R$ 8 mil": 5}
dim_aluno["ordem_faixa_renda"] = dim_aluno["faixa_renda_detalhada"].map(renda_order).fillna(99).astype("int8")

sexo_order = {"Feminino": 1, "Masculino": 2, "Não informado": 3, "X (não especificado)": 4}
dim_aluno["sexo"] = dim_aluno["sexo"].astype(str)
dim_aluno["ordem_sexo"] = dim_aluno["sexo"].map(sexo_order).fillna(99).astype("int8")

# Contagens por aluno (matriculas, avaliacoes, acessos, nota media clip)
mat_cnt = dim_aluno["id_aluno"].map(matriculas.groupby("id_aluno").size()).fillna(0).astype("int16")
aval_cnt = dim_aluno["id_aluno"].map(notas.groupby("id_aluno").size()).fillna(0).astype("int16")
ace_cnt = dim_aluno["id_aluno"].map(acessos.groupby("id_aluno").size()).fillna(0).astype("int16")
nota_clip_mean = dim_aluno["id_aluno"].map(notas.groupby("id_aluno")["nota_clip"].mean())

dim_aluno["qtd_matriculas"] = mat_cnt
dim_aluno["faixa_matriculas"] = np.where(mat_cnt >= 5, "5+", mat_cnt.astype(str))
dim_aluno["ordem_matriculas"] = np.minimum(mat_cnt, 5).astype("int8")

dim_aluno["qtd_avaliacoes"] = aval_cnt
dim_aluno["faixa_avaliacoes"] = np.where(aval_cnt >= 5, "5+", aval_cnt.astype(str))
dim_aluno["ordem_avaliacoes"] = np.minimum(aval_cnt, 5).astype("int8")

dim_aluno["qtd_acessos"] = ace_cnt
ace_cut = pd.cut(ace_cnt, bins=[-1, 0, 1, 3, 5, 10, 1000], labels=["0", "1", "2–3", "4–5", "6–10", "11+"])
ace_order = {"0": 1, "1": 2, "2–3": 3, "4–5": 4, "6–10": 5, "11+": 6}
dim_aluno["faixa_acessos"] = ace_cut.astype(str)
dim_aluno["ordem_acessos"] = dim_aluno["faixa_acessos"].map(ace_order).astype("int8")

dim_aluno["nota_media_clip"] = nota_clip_mean.round(4)
nota_bins = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10.01]
nota_labels = ["0–1", "1–2", "2–3", "3–4", "4–5", "5–6", "6–7", "7–8", "8–9", "9–10"]
nota_cut = pd.cut(nota_clip_mean, bins=nota_bins, labels=nota_labels, right=False)
nota_order = {k: i for i, k in enumerate(nota_labels, 1)}
dim_aluno["faixa_nota"] = nota_cut.astype(str).replace({"nan": None, "<NA>": None})
dim_aluno["ordem_nota"] = dim_aluno["faixa_nota"].map(nota_order).fillna(99).astype("int8")

dim_aluno["cobertura_digital"] = np.select(
    [
        (~dim_aluno["tem_acesso"]) & (~dim_aluno["tem_redes"]),
        (dim_aluno["tem_acesso"]) & (~dim_aluno["tem_redes"]),
        (dim_aluno["tem_acesso"]) & (dim_aluno["tem_redes"]),
        (~dim_aluno["tem_acesso"]) & (dim_aluno["tem_redes"]),
    ],
    ["Nenhuma das duas", "Só acessos", "Acessos e redes", "Só redes"],
    default="Nenhuma das duas",
)
dig_order = {"Nenhuma das duas": 1, "Só acessos": 2, "Acessos e redes": 3, "Só redes": 4}
dim_aluno["ordem_cobertura_digital"] = dim_aluno["cobertura_digital"].map(dig_order).astype("int8")

top12_set = set(dim_aluno.loc[~dim_aluno["flag_cidade_ausente"], "cidade"].value_counts().head(12).index)
dim_aluno["top12_cidade"] = dim_aluno["cidade"].isin(top12_set)

save_gold_table(dim_aluno, "dim_aluno", GOLD_DIR)

# 4. Tabela fato fato_matriculas
fato_matriculas = matriculas.merge(dim_curso, on=["faculdade", "curso"], how="left")
fato_matriculas["flag_evadido"] = (fato_matriculas["status"] == "Evadido").astype("int8")
fato_matriculas["flag_trancado"] = (fato_matriculas["status"] == "Trancado").astype("int8")
fato_matriculas["flag_ativo"] = (fato_matriculas["status"] == "Ativo").astype("int8")
mat_sit_order = {"Ativo": 1, "Evadido": 2, "Trancado": 3}
fato_matriculas["ordem_status"] = fato_matriculas["status"].astype(str).map(mat_sit_order).fillna(9).astype("int8")
save_gold_table(fato_matriculas, "fato_matriculas", GOLD_DIR)

# 5. Tabela fato fato_notas
fato_notas = notas[[
    "id_aluno", "disciplina", "data", "nota", "nota_bruta", "nota_clip", "flag_nota_invalida", "situacao_nota",
]].copy()
save_gold_table(fato_notas, "fato_notas", GOLD_DIR)

# 6. Tabela fato fato_financeiro
fato_financeiro = financeiro[[
    "id_lancamento", "id_aluno", "competencia", "valor_mensalidade",
    "valor_desconto", "valor_pago", "situacao_financeira", "situacao_financeira_raw",
    "forma_pagamento", "data_pagamento", "flag_mensalidade_negativa", "flag_pago_valor_zero",
]].copy()
fato_financeiro["flag_inadimplente"] = fato_financeiro["situacao_financeira"].isin(["EM_ATRASO", "PENDENTE"]).astype("int8")
fin_order = {"Pendente": 1, "Pago": 2, "Em atraso": 3, "Cancelado": 4, "Sem situação": 5}
fato_financeiro["ordem_situacao"] = fato_financeiro["situacao_financeira_raw"].astype(str).map(fin_order).fillna(9).astype("int8")
save_gold_table(fato_financeiro, "fato_financeiro", GOLD_DIR)

# 7. Visao consolidada fato_aluno_360 (agregacao por aluno matriculado para mineracao e ML)
agg_mat = fato_matriculas.groupby("id_aluno", as_index=False).agg(
    total_matriculas=("id_matricula", "count"),
    qtd_ativo=("flag_ativo", "sum"),
    qtd_evadido=("flag_evadido", "sum"),
    qtd_trancado=("flag_trancado", "sum"),
    faculdade_principal=("faculdade", "first"),
    curso_principal=("curso", "first"),
    semestre_principal=("semestre", "first"),
)
agg_mat["evadido_target"] = (agg_mat["qtd_evadido"] > 0).astype("int8")

agg_notas = fato_notas.groupby("id_aluno", as_index=False).agg(
    nota_media=("nota", "mean"),
    nota_mediana=("nota", "median"),
    qtd_notas_validas=("nota", "count"),
    qtd_notas_invalidas=("flag_nota_invalida", "sum"),
)

agg_acessos = acessos.groupby("id_aluno", as_index=False).agg(
    total_acessos=("tempo_logado", "count"),
    tempo_logado_total=("tempo_logado", "sum"),
    tempo_logado_medio=("tempo_logado", "mean"),
    paginas_total=("paginas", "sum"),
)

agg_redes = redes.groupby("id_aluno", as_index=False).agg(
    interacoes_total=("interacoes", "sum"),
    seguidores_medio=("seguidores", "mean"),
    seguindo_medio=("seguindo", "mean"),
)

agg_fin = fato_financeiro.groupby("id_aluno", as_index=False).agg(
    qtd_lancamentos_fin=("id_lancamento", "count"),
    valor_mensalidade_total=("valor_mensalidade", "sum"),
    valor_pago_total=("valor_pago", "sum"),
    qtd_inadimplente=("flag_inadimplente", "sum"),
)
agg_fin["taxa_inadimplencia"] = np.where(
    agg_fin["qtd_lancamentos_fin"] > 0,
    agg_fin["qtd_inadimplente"] / agg_fin["qtd_lancamentos_fin"],
    0.0,
)

agg_desl = deslocamento.groupby("id_aluno", as_index=False).agg(
    distancia_km_media=("distancia_km", "mean"),
    tempo_deslocamento_medio=("tempo_minutos", "mean"),
    custo_diario_medio=("custo_diario", "mean"),
)

fato_360 = (
    agg_mat
    .merge(dim_aluno[["id_aluno", "sexo", "idade", "faixa_etaria", "cidade", "renda_estimada", "faixa_renda"]], on="id_aluno", how="left")
    .merge(agg_notas, on="id_aluno", how="left")
    .merge(agg_acessos, on="id_aluno", how="left")
    .merge(agg_redes, on="id_aluno", how="left")
    .merge(agg_fin, on="id_aluno", how="left")
    .merge(agg_desl, on="id_aluno", how="left")
)

for zero_col in [
    "qtd_notas_validas", "qtd_notas_invalidas", "total_acessos",
    "tempo_logado_total", "paginas_total", "interacoes_total",
    "qtd_lancamentos_fin", "qtd_inadimplente",
]:
    fato_360[zero_col] = fato_360[zero_col].fillna(0)

save_gold_table(fato_360, "fato_aluno_360", GOLD_DIR)

resumo_star = {
    "dim_curso": len(dim_curso),
    "dim_calendario": len(dim_calendario),
    "dim_aluno": len(dim_aluno),
    "fato_matriculas": len(fato_matriculas),
    "fato_notas": len(fato_notas),
    "fato_financeiro": len(fato_financeiro),
    "fato_aluno_360": len(fato_360),
    "taxa_evasao_alunos_pct": round(100.0 * float(fato_360["evadido_target"].mean()), 2),
}
save_gold_json(resumo_star, "resumo_star_schema", GOLD_DIR)
print(json.dumps(resumo_star, ensure_ascii=False, indent=2))
