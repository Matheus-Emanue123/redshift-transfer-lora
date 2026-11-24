"""Prepara os conjuntos de treino, validação e teste para os modelos (MLP).

Decisões (ver o caderno e o texto do trabalho):
  - Entrada: os 5 brilhos cModel do HSC (g, r, i, z, y), sem correção de poeira.
  - Galáxias presentes em A e em B (mesmo hsc_object_id): retiradas do A e
    mantidas no B, com o redshift espectroscópico. Assim A e B não se repetem.
  - Divisão 80% treino, 10% validação, 10% teste, semente 42, em cada conjunto.
  - Combo = união das divisões de A e de B (as mesmas galáxias de teste em
    todos os modelos, sem vazamento entre treino e teste).
  - Opção A: frações de 10%, 25% e 50% do treino do B. As frações são
    encaixadas: a de 10% está dentro da de 25%, que está dentro da de 50%.

Entradas:
  data/processed/conjunto_a.csv
  data/processed/conjunto_b.csv
Saídas, em data/modelo/:
  a_treino.csv, a_val.csv, a_teste.csv
  b_treino.csv, b_val.csv, b_teste.csv
  combo_treino.csv, combo_val.csv, combo_teste.csv
  b_treino_10.csv, b_treino_25.csv, b_treino_50.csv
Colunas: hsc_object_id, g, r, i, z, y, redshift, origem (A ou B)

Uso (da raiz do repositório):
  python src/preparar_dados.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

SEMENTE = 42
FRACOES = [10, 25, 50]
SAIDA = Path("data/modelo")
BANDAS = ["g", "r", "i", "z", "y"]


def ler(caminho, origem):
    df = pd.read_csv(caminho)
    out = pd.DataFrame({"hsc_object_id": df["hsc_object_id"].astype("int64")})
    for b in BANDAS:
        out[b] = df[f"hsc_{b}_cmodel_mag"]
    out["redshift"] = df["redshift"]
    out["origem"] = origem
    return out


def dividir(df, semente):
    """Embaralha e divide em 80/10/10."""
    df = df.sample(frac=1, random_state=semente).reset_index(drop=True)
    n = len(df)
    n_treino, n_val = int(0.8 * n), int(0.1 * n)
    return df[:n_treino], df[n_treino:n_treino + n_val], df[n_treino + n_val:]


def main():
    a = ler("data/processed/conjunto_a.csv", "A")
    b = ler("data/processed/conjunto_b.csv", "B")
    print(f"conjunto A: {len(a)} galáxias | conjunto B: {len(b)} galáxias")

    # galáxias nos dois conjuntos: ficam só no B
    comuns = a["hsc_object_id"].isin(b["hsc_object_id"])
    print(f"galáxias nos dois conjuntos (retiradas do A): {int(comuns.sum())}")
    a = a[~comuns]

    # conferência: nenhum valor faltando na entrada nem no redshift
    for nome, df in [("A", a), ("B", b)]:
        faltando = int(df[BANDAS + ["redshift"]].isna().any(axis=1).sum())
        if faltando:
            print(f"  ATENÇÃO: {faltando} linhas do {nome} com valor faltando (serão removidas)")
    a = a.dropna(subset=BANDAS + ["redshift"])
    b = b.dropna(subset=BANDAS + ["redshift"])

    a_tr, a_va, a_te = dividir(a, SEMENTE)
    b_tr, b_va, b_te = dividir(b, SEMENTE)

    SAIDA.mkdir(parents=True, exist_ok=True)
    partes = {
        "a_treino": a_tr, "a_val": a_va, "a_teste": a_te,
        "b_treino": b_tr, "b_val": b_va, "b_teste": b_te,
        "combo_treino": pd.concat([a_tr, b_tr], ignore_index=True),
        "combo_val": pd.concat([a_va, b_va], ignore_index=True),
        "combo_teste": pd.concat([a_te, b_te], ignore_index=True),
    }
    # frações do treino do B (b_tr já está embaralhado: os primeiros N formam frações encaixadas)
    for f in FRACOES:
        partes[f"b_treino_{f}"] = b_tr[: int(len(b_tr) * f / 100)]

    print(f"\n{'arquivo':<18}{'galáxias':>10}{'z mediano':>12}")
    for nome, df in partes.items():
        df.to_csv(SAIDA / f"{nome}.csv", index=False)
        print(f"{nome:<18}{len(df):>10}{df['redshift'].median():>12.3f}")
    print(f"\nsalvo em {SAIDA}/")


if __name__ == "__main__":
    main()