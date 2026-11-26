"""Camada Gold: amostragem, balanceamento, classificacao de evasao, regressao linear e clustering K-Means."""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
    silhouette_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

# Garante acesso ao pacote utils na pasta notebooks/
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

fato_360 = pd.read_parquet(GOLD_DIR / "fato_aluno_360.parquet")

# 1. Comparacao das 4 tecnicas de amostragem classica (aleatoria, estratificada, sistematica e clusters)
amostra_aleatoria = fato_360.sample(n=30000, random_state=42)
_, amostra_estratificada = train_test_split(
    fato_360, test_size=30000, stratify=fato_360["evadido_target"], random_state=42
)
step = max(1, len(fato_360) // 30000)
amostra_sistematica = fato_360.iloc[::step].head(30000)
top_clusters = fato_360["faculdade_principal"].value_counts().head(2).index.tolist()
amostra_clusters = fato_360[fato_360["faculdade_principal"].isin(top_clusters)].sample(n=30000, random_state=42)

sampling_summary = [
    {"tecnica": "Amostragem Aleatória Simples", "n": len(amostra_aleatoria), "taxa_evasao_pct": round(100 * float(amostra_aleatoria["evadido_target"].mean()), 2)},
    {"tecnica": "Amostragem Estratificada", "n": len(amostra_estratificada), "taxa_evasao_pct": round(100 * float(amostra_estratificada["evadido_target"].mean()), 2)},
    {"tecnica": "Amostragem Sistemática", "n": len(amostra_sistematica), "taxa_evasao_pct": round(100 * float(amostra_sistematica["evadido_target"].mean()), 2)},
    {"tecnica": "Amostragem por Clusters", "n": len(amostra_clusters), "taxa_evasao_pct": round(100 * float(amostra_clusters["evadido_target"].mean()), 2)},
]
save_gold_table(pd.DataFrame(sampling_summary), "amostragem_classica", GOLD_DIR)

# 2. Preparacao de features na amostra estratificada (30k) para treino agil e posterior escoragem em toda a base
features = [
    "idade", "renda_estimada", "nota_media", "tempo_logado_medio",
    "paginas_total", "interacoes_total", "taxa_inadimplencia", "total_matriculas",
]
df_ml = amostra_estratificada.copy()
medians = {c: float(fato_360[c].median()) for c in features}
for c in features:
    df_ml[c] = df_ml[c].fillna(medians[c])

X = df_ml[features]
y = df_ml["evadido_target"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, stratify=y, random_state=42)
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc = scaler.transform(X_test)

# 3. Balanceamento (Undersampling, Oversampling, SMOTE) e avaliacao dos classificadores
samplers = {
    "Original (Desbalanceado)": (X_train_sc, y_train),
    "Undersampling": RandomUnderSampler(random_state=42).fit_resample(X_train_sc, y_train),
    "Oversampling": RandomOverSampler(random_state=42).fit_resample(X_train_sc, y_train),
    "SMOTE": SMOTE(random_state=42).fit_resample(X_train_sc, y_train),
}

models_cfg = {
    "LogisticRegression": lambda: LogisticRegression(max_iter=500, random_state=42),
    "DecisionTree": lambda: DecisionTreeClassifier(max_depth=6, random_state=42),
    "RandomForest": lambda: RandomForestClassifier(n_estimators=80, max_depth=8, n_jobs=-1, random_state=42),
}

eval_rows = []
best_model = None
best_f1 = -1.0
best_name = ""

for samp_name, (Xs, ys) in samplers.items():
    for mod_name, mod_fn in models_cfg.items():
        clf = mod_fn()
        clf.fit(Xs, ys)
        y_pred = clf.predict(X_test_sc)
        y_prob = clf.predict_proba(X_test_sc)[:, 1]
        f1 = float(f1_score(y_test, y_pred, zero_division=0))
        auc = float(roc_auc_score(y_test, y_prob))
        eval_rows.append({
            "balanceamento": samp_name,
            "modelo": mod_name,
            "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
            "precision": round(float(precision_score(y_test, y_pred, zero_division=0)), 4),
            "recall": round(float(recall_score(y_test, y_pred, zero_division=0)), 4),
            "f1_score": round(f1, 4),
            "roc_auc": round(auc, 4),
        })
        if f1 > best_f1:
            best_f1 = f1
            best_model = clf
            best_name = f"{mod_name} + {samp_name}"

# Validacao cruzada (5-fold) no algoritmo base RandomForest
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(
    RandomForestClassifier(n_estimators=60, max_depth=8, n_jobs=-1, random_state=42),
    X_train_sc,
    y_train,
    cv=skf,
    scoring="roc_auc",
)

# Bootstrap (50 iteracoes no conjunto de teste) para estimar intervalo de confianca de 95% do F1
rng = np.random.default_rng(42)
boot_f1 = []
y_test_arr = y_test.to_numpy()
best_pred = best_model.predict(X_test_sc)
for _ in range(50):
    idx = rng.integers(0, len(y_test_arr), size=len(y_test_arr))
    boot_f1.append(f1_score(y_test_arr[idx], best_pred[idx], zero_division=0))

metricas_df = pd.DataFrame(eval_rows).sort_values("f1_score", ascending=False)
save_gold_table(metricas_df, "metricas_modelos", GOLD_DIR)

# 4. Importancia das variaveis do melhor modelo e regressao linear para predicao de nota_media
if hasattr(best_model, "feature_importances_"):
    fi_df = pd.DataFrame({"feature": features, "importancia": best_model.feature_importances_}).sort_values("importancia", ascending=False)
else:
    fi_df = pd.DataFrame({"feature": features, "importancia": np.abs(best_model.coef_[0])}).sort_values("importancia", ascending=False)
save_gold_table(fi_df, "feature_importance", GOLD_DIR)

reg_cols = ["tempo_logado_medio", "paginas_total", "interacoes_total", "renda_estimada", "idade"]
reg_df = df_ml[reg_cols + ["nota_media"]].dropna()
lr = LinearRegression().fit(reg_df[reg_cols], reg_df["nota_media"])
nota_pred = lr.predict(reg_df[reg_cols])
reg_r2 = round(float(r2_score(reg_df["nota_media"], nota_pred)), 4)
reg_rmse = round(float(np.sqrt(mean_squared_error(reg_df["nota_media"], nota_pred))), 4)

# 5. Agrupamento K-Means (k=4) com avaliacao via Silhouette Score
cluster_feats = ["nota_media", "tempo_logado_medio", "interacoes_total", "renda_estimada"]
X_full_imp = fato_360[features].copy()
for c in features:
    X_full_imp[c] = X_full_imp[c].fillna(medians[c])

sc_clust = StandardScaler()
X_clust_sample = sc_clust.fit_transform(df_ml[cluster_feats])
kmeans = KMeans(n_clusters=4, random_state=42, n_init=10).fit(X_clust_sample)
sil_score = round(float(silhouette_score(X_clust_sample[:10000], kmeans.labels_[:10000])), 4)

# Aplica probabilidade de evasao, faixa de risco e cluster na base completa fato_aluno_360
X_full_sc = scaler.transform(X_full_imp)
fato_360["prob_evasao"] = np.round(best_model.predict_proba(X_full_sc)[:, 1], 4)
fato_360["faixa_risco_evasao"] = pd.cut(
    fato_360["prob_evasao"],
    bins=[-0.01, 0.35, 0.65, 1.01],
    labels=["Baixo Risco", "Médio Risco", "Alto Risco"],
).astype("category")

X_full_clust = sc_clust.transform(X_full_imp[cluster_feats])
fato_360["cluster_id"] = kmeans.predict(X_full_clust)
cluster_perfil = (
    fato_360.groupby("cluster_id", as_index=False)
    .agg(
        alunos=("id_aluno", "count"),
        taxa_evasao_pct=("evadido_target", lambda s: round(100 * float(s.mean()), 2)),
        nota_media=("nota_media", "mean"),
        tempo_logado_medio=("tempo_logado_medio", "mean"),
        interacoes_media=("interacoes_total", "mean"),
        renda_media=("renda_estimada", "mean"),
    )
)
save_gold_table(cluster_perfil, "perfil_clusters", GOLD_DIR)
save_gold_table(fato_360, "fato_aluno_360", GOLD_DIR)

# Salva resumo consolidado das metricas de ML na camada Gold
ml_summary = {
    "melhor_modelo": best_name,
    "melhor_f1_teste": round(best_f1, 4),
    "cv_5fold_roc_auc_mean": round(float(cv_scores.mean()), 4),
    "bootstrap_f1_ci95": [round(float(np.percentile(boot_f1, 2.5)), 4), round(float(np.percentile(boot_f1, 97.5)), 4)],
    "regressao_linear_nota_r2": reg_r2,
    "regressao_linear_nota_rmse": reg_rmse,
    "kmeans_k": 4,
    "kmeans_silhouette_score": sil_score,
}
save_gold_json(ml_summary, "resumo_ml_gold", GOLD_DIR)
print(json.dumps(ml_summary, ensure_ascii=False, indent=2))
print(metricas_df.head(6).to_string(index=False))
