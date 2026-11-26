"""Camada Silver: limpeza, padronizacao, validacao, outliers, binning e anonimizacao LGPD de alunos."""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Garante acesso ao pacote utils na pasta notebooks/
notebooks_dir = Path(__file__).resolve().parent.parent
if (notebooks_dir / "utils").exists() and str(notebooks_dir) not in sys.path:
    sys.path.insert(0, str(notebooks_dir))

from utils.silver_utils import (
    anonymize_name_lgpd,
    detect_outliers_iqr,
    detect_outliers_zscore,
    fix_mojibake_text,
    get_silver_paths,
    load_bronze,
    normalize_id_aluno,
    parse_br_float,
    parse_date_col,
    save_silver_outputs,
)

ROOT_DIR, BRONZE_DIR, SILVER_DIR = get_silver_paths()

df = load_bronze("alunos", BRONZE_DIR)
rows_in = len(df)

# 1. Padronizacao de chaves e datas
df["id_aluno"] = normalize_id_aluno(df["id_aluno"])
df["data_nascimento"] = parse_date_col(df["data_nascimento"])

# 2. Padronizacao e recuperacao deterministica de sexo pelo primeiro nome antes da anonimizacao LGPD
sexo_raw_map = {
    "f": "Feminino",
    "fem": "Feminino",
    "feminino": "Feminino",
    "m": "Masculino",
    "masculino": "Masculino",
    "x": "X (não especificado)",
}
sexo_clean = df["sexo"].astype(str).str.strip().str.lower().map(sexo_raw_map)
first_name = df["nome"].astype(str).str.strip().str.split().str[0].str.lower()
known_gender = sexo_clean.isin(["Feminino", "Masculino"])
name_to_gender = (
    pd.DataFrame({"fn": first_name[known_gender], "sx": sexo_clean[known_gender]})
    .groupby("fn")["sx"]
    .agg(lambda s: s.mode().iloc[0] if len(s.mode()) == 1 else pd.NA)
    .dropna()
    .to_dict()
)
miss_sexo = sexo_clean.isna()
sexo_recovered = sexo_clean.where(~miss_sexo, first_name.map(name_to_gender))
df["flag_sexo_ausente"] = sexo_recovered.isna()
df["sexo"] = sexo_recovered.fillna("Não informado").astype("category")

# 3. LGPD: anonimizacao de nome (pseudonimo SHA-256) e remocao do dado pessoal em claro
df["nome_anonimizado"] = anonymize_name_lgpd(df["nome"], df["id_aluno"])
df = df.drop(columns=["nome"])

# 4. Limpeza textual, correcao de mojibake e ortografia em cidade
df["cidade"], mojibake_cidade = fix_mojibake_text(df["cidade"])
cidade_norm = df["cidade"].str.title().replace({
    "Matto": "Matão",
    "Sao Paulo": "São Paulo",
    "Desconhecida": "Não Informada",
    "": "Não Informada",
})
df["flag_cidade_ausente"] = cidade_norm == "Não Informada"
df["cidade"] = cidade_norm
cidade_imputadas = int(df["flag_cidade_ausente"].sum())

# 5. Recuperacao textual/temporal de idade, validacao, imputacao por mediana e binning
idade_txt = (
    df["idade"].astype(str).str.strip().str.lower()
    .replace({"vinte": "20"})
    .str.replace(" anos", "", regex=False)
)
idade_num = pd.to_numeric(idade_txt, errors="coerce")
idade_dn = (pd.Timestamp("2025-12-31") - df["data_nascimento"]).dt.days / 365.25
idade_bad = idade_num.isna() | (idade_num < 10) | (idade_num > 100)
idade_rec = idade_num.where(~idade_bad, idade_dn.floordiv(1))
idade_invalida_mask = idade_rec.isna() | (idade_rec < 10) | (idade_rec > 100)
df["flag_idade_ausente"] = idade_invalida_mask
df["flag_idade_imputada"] = idade_invalida_mask
idade_mediana = float(idade_rec.loc[~idade_invalida_mask].median())
df["idade"] = idade_rec.where(~idade_invalida_mask, idade_mediana).round().astype("Int64")

df["faixa_etaria_detalhada"] = pd.cut(
    idade_rec.where(~idade_invalida_mask, np.nan),
    bins=[-1, 17, 29, 39, 49, 59, 200],
    labels=["Até 17", "18–29", "30–39", "40–49", "50–59", "60+"],
).astype("category")
df["faixa_etaria"] = pd.cut(
    df["idade"].astype(float),
    bins=[0, 20, 25, 35, 50, 120],
    labels=["<=20", "21-25", "26-35", "36-50", ">50"],
).astype("category")

# 6. Padronizacao decimal BR de renda_estimada, imputacao por mediana, outliers e binning
renda_num = parse_br_float(df["renda_estimada"])
renda_invalida_mask = renda_num.isna() | (renda_num < 0)
df["flag_renda_ausente"] = renda_invalida_mask
df["flag_renda_imputada"] = renda_invalida_mask
renda_mediana = float(renda_num.loc[~renda_invalida_mask].median())
df["renda_estimada"] = renda_num.where(~renda_invalida_mask, renda_mediana)
df["flag_renda_outlier_iqr"], iqr_info = detect_outliers_iqr(df["renda_estimada"])
df["flag_renda_outlier_zscore"], z_count = detect_outliers_zscore(df["renda_estimada"])
df["faixa_renda_detalhada"] = pd.cut(
    renda_num.where(~renda_invalida_mask, np.nan),
    bins=[-0.01, 2000, 4000, 6000, 8000, 1e9],
    labels=["Até R$ 2 mil", "R$ 2–4 mil", "R$ 4–6 mil", "R$ 6–8 mil", "Acima de R$ 8 mil"],
).astype("category")
df["faixa_renda"] = pd.qcut(
    df["renda_estimada"],
    q=4,
    labels=["Q1 (Baixa)", "Q2 (Média-Baixa)", "Q3 (Média-Alta)", "Q4 (Alta)"],
).astype("category")

# 7. Remocao de duplicatas pela chave primaria id_aluno
dups_removed = int(df.duplicated(subset=["id_aluno"]).sum())
df = df.drop_duplicates(subset=["id_aluno"]).reset_index(drop=True)

# Registra log de transformacoes e salva tabela Silver em Parquet
log = {
    "dataset": "alunos",
    "rows_in": rows_in,
    "rows_out": int(len(df)),
    "duplicates_removed": dups_removed,
    "lgpd_nome_anonymized": True,
    "mojibake_fixed_cidade": mojibake_cidade,
    "sexo_ausente_apos_recuperacao": int(df["flag_sexo_ausente"].sum()),
    "cidade_ausente_apos_padronizacao": cidade_imputadas,
    "idade_ausente_apos_recuperacao": int(df["flag_idade_ausente"].sum()),
    "idade_mediana_valor": idade_mediana,
    "renda_ausente_apos_padronizacao_br": int(df["flag_renda_ausente"].sum()),
    "renda_mediana_valor": round(renda_mediana, 2),
    "renda_outliers_iqr": iqr_info,
    "renda_outliers_zscore": z_count,
}
save_silver_outputs(df, log, "alunos", SILVER_DIR)
print(json.dumps(log, ensure_ascii=False, indent=2))
print(df.head(5))
