# Trabalho Final — Mineração de Dados (Arquitetura Medallion + Power BI)

Projeto desenvolvido para a disciplina de **Mineração de Dados** da **FATEC Matão-SP** (Prof. Rodrigo Vidrich), aplicando o processo **KDD / CRISP-DM** em uma base acadêmica com **1.000.000 de alunos** e **8 fontes de dados**, utilizando **Arquitetura Medallion (`Bronze → Silver → Gold`)** em Python e visualização executiva no **Power BI (`.pbip`)**.

## Integrantes do Grupo
- Jerônimo Barbieri Junior
- Luan Chierato Oliveira
- Miguel Miranda Morandini
- Victor Hugo Malipense Testi

---

## Estrutura do Projeto

```text
trabalho-final-mineacao-de-dados/
├── notebooks/
│   ├── utils/                        # Funções reutilizáveis por camada
│   │   ├── bronze_utils.py
│   │   ├── silver_utils.py
│   │   └── gold_utils.py
│   ├── 01_bronze/                    # Ingestão bruta, auditoria de delimitadores e profiling
│   │   ├── 01_alunos.ipynb ... 08_frequencia.ipynb
│   │   └── 09_consolidado_diagnostico.ipynb
│   ├── 02_silver/                    # Limpeza, padronização, outliers, regras de negócio e LGPD
│   │   ├── 01_alunos.ipynb ... 08_frequencia.ipynb
│   │   └── 09_consolidado_qualidade.ipynb
│   └── 03_gold/                      # Star Schema, estatística, amostragem, ML, associação e ARS
│       ├── 01_star_schema_powerbi.ipynb
│       ├── 02_estatistica_descritiva_e_correlacao.ipynb
│       ├── 03_amostragem_e_modelagem_ml.ipynb
│       └── 04_regras_associacao_e_ars.ipynb
├── data/                             # Saídas locais geradas pelo pipeline (resumos versionados)
│   ├── bronze/
│   ├── silver/
│   └── gold/
├── powerbi/                          # Projeto Power BI (.pbip) conectado à camada Gold
│   ├── TrabalhoFinal_Mineracao.pbip
│   ├── TrabalhoFinal_Mineracao.SemanticModel/
│   └── TrabalhoFinal_Mineracao.Report/
├── requirements.txt
└── README.md
```

---

## Pré-requisitos

- **Python 3.12+**
- **Power BI Desktop** (para abrir o projeto `.pbip`)
- Pasta `Bases TRabalho Final/` na raiz do repositório contendo os 8 arquivos originais da disciplina (não versionados no Git devido ao tamanho).

---

## Como Executar o Projeto (Passo a Passo)

### 1. Clonar o repositório e configurar o ambiente Python

No PowerShell (Windows), dentro da pasta do projeto:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 2. Executar a Camada Bronze (`notebooks/01_bronze/`)
Responsável pela ingestão bruta (*Staging Area*), preservando todos os dados originais em `data/bronze/*.parquet`, auditando delimitadores mistos (`|` e `;` no financeiro) e gerando o diagnóstico de qualidade (`resumo_bronze.csv`).

```powershell
Get-ChildItem "notebooks\01_bronze\*.ipynb" | Sort-Object Name | ForEach-Object {
    jupyter nbconvert --to notebook --execute --inplace $_.FullName
}
```

### 3. Executar a Camada Silver (`notebooks/02_silver/`)
Aplica as 8 técnicas de limpeza de dados: tratamento de faltantes (remoção + imputação por mediana/moda), padronização de formatos (`sexo`, vírgula decimal, datas), remoção de duplicatas, correção de *mojibake*, detecção de outliers (**IQR** e **Z-Score**), *binning*, validação de regras de negócio (`idade` `[15, 90]`, `nota` `[0, 10]`) e **anonimização LGPD** de `nome` (`SHA-256`).

```powershell
Get-ChildItem "notebooks\02_silver\*.ipynb" | Sort-Object Name | ForEach-Object {
    jupyter nbconvert --to notebook --execute --inplace $_.FullName
}
```

### 4. Executar a Camada Gold (`notebooks/03_gold/`)
Constrói o **Star Schema** (`dim_aluno`, `dim_curso`, `dim_calendario`, `fato_matriculas`, `fato_notas`, `fato_financeiro`, `fato_aluno_360`) e executa:
- **Estatística Descritiva & Correlação:** Média, Mediana, Moda, Amplitude, Variância, Desvio Padrão, CV, Quartis/Decis/Percentis, Covariância, **Pearson** e **Spearman**.
- **Amostragem Clássica & Específica:** Aleatória Simples, Estratificada, Sistemática, Clusters, **Undersampling**, **Oversampling**, **SMOTE**, **Bootstrap** e **Cross-Validation (5-fold)**.
- **Mineração de Dados:** Classificação de evasão (`LogisticRegression`, `DecisionTree`, `RandomForest`), Regressão Linear, Clustering (**K-Means** + Silhouette), Regras de Associação (Suporte, Confiança, Lift) e ARS (`NetworkX`).

```powershell
Get-ChildItem "notebooks\03_gold\*.ipynb" | Sort-Object Name | ForEach-Object {
    jupyter nbconvert --to notebook --execute --inplace $_.FullName
}
```

### 5. Abrir o Dashboard no Power BI Desktop
1. Certifique-se de que a Camada Gold foi executada e gerou os CSVs em `data/gold/`.
2. Dê duplo clique em `powerbi/TrabalhoFinal_Mineracao.pbip`.
3. Caso o caminho local dos CSVs seja diferente do padrão (`C:\Projects\trabalho-final-mineacao-de-dados\data\gold\`), atualize o caminho da fonte em **Transformar Dados → Configurações da Fonte de Dados** e clique em **Atualizar**.

---

## Tecnologias Utilizadas
- **Python 3.12+** (`pandas`, `pyarrow`, `scikit-learn`, `imbalanced-learn`, `networkx`, `scipy`)
- **Jupyter Notebooks** (arquitetura modular em camadas Bronze, Silver e Gold)
- **Power BI Desktop (`.pbip`)** para modelagem semântica e visualização executiva
- **Git / GitHub** para versionamento
