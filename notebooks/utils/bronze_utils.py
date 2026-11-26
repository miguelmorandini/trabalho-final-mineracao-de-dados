"""Funções reutilizáveis de ingestão bruta e diagnóstico para a Camada Bronze."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


def get_project_paths() -> tuple[Path, Path, Path]:
    """Resolve os diretórios raiz, origem bruta e destino Bronze a partir do CWD atual."""
    cwd = Path.cwd().resolve()
    for candidate in [cwd, cwd.parent, cwd.parent.parent]:
        if (candidate / "Bases TRabalho Final").exists():
            raw_dir = candidate / "Bases TRabalho Final"
            bronze_dir = candidate / "data" / "bronze"
            bronze_dir.mkdir(parents=True, exist_ok=True)
            return candidate, raw_dir, bronze_dir
    raise FileNotFoundError(
        "Não foi possível localizar a pasta 'Bases TRabalho Final' a partir de "
        f"{cwd}"
    )


def utc_timestamp() -> str:
    """Retorna timestamp UTC ISO-8601 para metadados de linhagem."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _read_mixed_delimiter_file(file_path: Path) -> tuple[pd.DataFrame, dict[str, int]]:
    """Lê arquivo texto com delimitadores mistos ('|' e ';') preservando todas as linhas."""
    rows: list[list[str]] = []
    raw_delims: list[str] = []
    delimiter_counts: dict[str, int] = {}

    with open(file_path, "r", encoding="utf-8", errors="replace") as f:
        header_line = f.readline().rstrip("\r\n")
        header_sep = "|" if "|" in header_line else ";"
        columns = header_line.split(header_sep)
        n_cols = len(columns)

        for line in f:
            clean_line = line.rstrip("\r\n")
            if not clean_line:
                continue
            sep_used = ";" if (clean_line.count(";") == n_cols - 1 and "|" not in clean_line) else "|"
            parts = clean_line.split(sep_used)
            if len(parts) != n_cols:
                alt_sep = ";" if sep_used == "|" else "|"
                alt_parts = clean_line.split(alt_sep)
                if len(alt_parts) == n_cols:
                    parts = alt_parts
                    sep_used = alt_sep
            rows.append(parts[:n_cols] + [""] * max(0, n_cols - len(parts)))
            raw_delims.append(sep_used)
            delimiter_counts[sep_used] = delimiter_counts.get(sep_used, 0) + 1

    df = pd.DataFrame(rows, columns=columns, dtype=str)
    df["_bronze_raw_delimiter"] = raw_delims
    return df, delimiter_counts


def ingest_raw_file(
    dataset_name: str,
    filename: str,
    sep: str = "|",
    mixed_sep: bool = False,
    raw_dir: Path | None = None,
    ingested_at: str | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Ingere um arquivo bruto como string, adicionando metadados de linhagem Bronze."""
    if raw_dir is None:
        _, raw_dir, _ = get_project_paths()
    if ingested_at is None:
        ingested_at = utc_timestamp()

    file_path = raw_dir / filename
    if not file_path.exists():
        raise FileNotFoundError(f"Arquivo bruto não encontrado: {file_path}")

    size_mb = round(file_path.stat().st_size / (1024 * 1024), 4)

    if mixed_sep:
        df, delimiter_counts = _read_mixed_delimiter_file(file_path)
    else:
        df = pd.read_csv(
            file_path,
            sep=sep,
            dtype=str,
            keep_default_na=False,
            encoding="utf-8",
            encoding_errors="replace",
        )
        delimiter_counts = {sep: int(len(df))}

    df["_bronze_source_file"] = filename
    df["_bronze_ingested_at"] = ingested_at

    meta = {
        "dataset": dataset_name,
        "source_file": filename,
        "size_mb": size_mb,
        "rows": int(len(df)),
        "columns": [c for c in df.columns if not c.startswith("_bronze_")],
        "delimiter_counts": delimiter_counts,
        "ingested_at": ingested_at,
    }
    return df, meta


def _compute_dataset_anomalies(
    dataset_name: str, df: pd.DataFrame, meta: dict
) -> dict:
    """Levanta anomalias específicas de cada tabela para orientar a camada Silver."""
    anomalies: dict = {}
    if dataset_name == "alunos":
        anomalies["sexo_distribuicao"] = df["sexo"].value_counts(dropna=False).to_dict()
        idade_num = pd.to_numeric(df["idade"], errors="coerce")
        anomalies["idade_menor_15_ou_maior_90"] = int(((idade_num < 15) | (idade_num > 90)).sum())
        anomalies["renda_com_virgula"] = int(df["renda_estimada"].str.contains(",", regex=False).sum())
        anomalies["mojibake_cidade_ou_nome"] = int(
            (df["cidade"].str.contains("Ã", regex=False) | df["nome"].str.contains("Ã", regex=False)).sum()
        )
    elif dataset_name == "notas":
        nota_num = pd.to_numeric(df["nota"].str.replace(",", ".", regex=False), errors="coerce")
        anomalies["nota_menor_0"] = int((nota_num < 0).sum())
        anomalies["nota_maior_10"] = int((nota_num > 10).sum())
    elif dataset_name == "matriculas":
        anomalies["status_distribuicao"] = df["status"].value_counts(dropna=False).to_dict()
    elif dataset_name == "financeiro":
        anomalies["situacao_distribuicao"] = df["situacao_financeira"].value_counts(dropna=False).to_dict()
        anomalies["delimitadores_detectados"] = meta["delimiter_counts"]
    elif dataset_name == "frequencia":
        anomalies["situacao_distribuicao"] = df["situacao"].value_counts(dropna=False).to_dict()
        anomalies["escopo_observacao"] = (
            "Micro-amostra complementar (20 linhas / 9 alunos) — não extrapolar para 1M."
        )
    return anomalies


def profile_bronze_table(
    df: pd.DataFrame, meta: dict, alunos_ids: set[str] | None = None
) -> dict:
    """Gera diagnóstico quantitativo de qualidade para uma tabela Bronze."""
    dataset_name = meta["dataset"]
    biz_cols = [c for c in df.columns if not c.startswith("_bronze_")]
    empty_counts = {c: int((df[c].str.strip() == "").sum()) for c in biz_cols}
    exact_dups = int(df.duplicated(subset=biz_cols).sum())

    unique_ids: set[str] = set()
    coverage_pct: float | None = None
    if "id_aluno" in df.columns:
        valid_id = df["id_aluno"].str.strip()
        non_empty = valid_id[valid_id != ""].str.lstrip("0").replace("", "0")
        unique_ids = set(non_empty)
        if alunos_ids is not None and unique_ids:
            coverage_pct = round(100.0 * len(unique_ids & alunos_ids) / len(unique_ids), 2)

    return {
        "metadata": meta,
        "unique_id_aluno": len(unique_ids),
        "id_aluno_match_alunos_pct": coverage_pct,
        "exact_duplicates": exact_dups,
        "empty_counts": empty_counts,
        "total_empty_fields": int(sum(empty_counts.values())),
        "anomalies": _compute_dataset_anomalies(dataset_name, df, meta),
    }


def save_bronze_outputs(
    df: pd.DataFrame, profile: dict, dataset_name: str, bronze_dir: Path
) -> tuple[Path, Path]:
    """Salva o snapshot Parquet Bronze e o JSON individual de qualidade."""
    parquet_path = bronze_dir / f"{dataset_name}.parquet"
    profile_path = bronze_dir / f"{dataset_name}_profile.json"
    df.to_parquet(parquet_path, index=False, compression="snappy")
    with open(profile_path, "w", encoding="utf-8") as f:
        json.dump(profile, f, ensure_ascii=False, indent=2)
    return parquet_path, profile_path
