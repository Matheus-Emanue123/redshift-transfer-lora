"""Limpeza do conjunto B (espectroscópico), seguindo a seção 3.2 do GalaxiesML.

Critérios, na ordem do artigo:
  1. 0,01 < redshift < 4
  2. erro do redshift < 0,005 / (1 + redshift)
  3. brilho cModel entre 0 e 50 mag nos 5 filtros
  4. duplicatas: mesmo specz_name; depois, mesmo object_id com specz_name
     diferente. O GalaxiesML mantém a primeira linha. Decisão nossa: antes de
     remover, ordenamos as linhas para que as com isprimary verdadeiro venham
     primeiro. Assim, a linha mantida é a medida principal do HSC.

O artigo se contradiz sobre specz_flag_homogeneous: a seção 3.1 diz que foi
exigido, mas a consulta SQL do apêndice não aplica esse filtro. O script mostra
a contagem final dos dois jeitos. O que der mais perto de 286.401 (número final
do artigo) indica o que eles fizeram. A escolha fica em USAR_FLAG_HOMOGENEO.

Entrada: data/raw/hsc_pdr2_wide_specz.csv.gz
Saída:   data/processed/conjunto_b.csv

Uso (da raiz do repositório):
  python src/limpeza_b.py
"""
import gzip
from pathlib import Path

import pandas as pd

ENTRADA = Path("data/raw/hsc_pdr2_wide_specz.csv.gz")
SAIDA = Path("data/processed/conjunto_b.csv")
USAR_FLAG_HOMOGENEO = True

B = ["g", "r", "i", "z", "y"]
CMODEL = [f"{b}_cmodel_mag" for b in B]


def ler_hsc(caminho):
    """Lê o arquivo do HSC: o cabeçalho real é a linha que começa com '# object_id'."""
    with gzip.open(caminho, "rt", encoding="utf-8") as f:
        for linha in f:
            if linha.startswith("# object_id"):
                nomes = [c.strip() for c in linha[2:].split(",")]
                break
        else:
            raise ValueError("cabeçalho '# object_id' não encontrado")
        df = pd.read_csv(f, names=nomes, header=None)
    return df[df["object_id"].notna()]


def verdadeiro(serie):
    return serie.astype(str).str.lower().isin(["t", "true", "1"])


def main():
    df = ler_hsc(ENTRADA)
    print(f"{'entrada (consulta SQL)':<50} {len(df):>9}   (GalaxiesML: 801.246)")

    def registrar(nome, mascara):
        nonlocal df
        df = df[mascara].copy()
        print(f"{nome:<50} {len(df):>9}")

    z = df["specz_redshift"]
    registrar("1. 0,01 < redshift < 4", (z > 0.01) & (z < 4))

    z, err = df["specz_redshift"], df["specz_redshift_err"]
    registrar("2. erro < 0,005 / (1 + z)", err < 0.005 / (1 + z))

    v = df[CMODEL]
    registrar("3. cModel entre 0 e 50 mag nos 5 filtros", ((v > 0) & (v < 50)).all(axis=1))

    # linhas com isprimary verdadeiro primeiro (ordenação estável mantém o resto da ordem)
    df["_primario"] = verdadeiro(df["isprimary"])
    df = df.sort_values("_primario", ascending=False, kind="stable")
    registrar("4a. duplicatas por specz_name", ~df.duplicated("specz_name", keep="first"))
    registrar("4b. duplicatas por object_id", ~df.duplicated("object_id", keep="first"))

    com_flag = verdadeiro(df["specz_flag_homogeneous"])
    print(f"{'   final SEM exigir specz_flag_homogeneous':<50} {len(df):>9}")
    print(f"{'   final EXIGINDO specz_flag_homogeneous':<50} {int(com_flag.sum()):>9}")
    print(f"{'   (GalaxiesML final: 286.401)':<50}")
    if USAR_FLAG_HOMOGENEO:
        registrar("5. specz_flag_homogeneous = verdadeiro", com_flag)

    # mesmos nomes de coluna do conjunto A, para juntar depois no Combo
    colunas = (["object_id", "ra", "dec"] + CMODEL
               + [f"{b}_cmodel_magsigma" for b in B] + [f"a_{b}" for b in B])
    final = df[colunas + ["specz_name", "specz_redshift", "specz_redshift_err"]]
    final = final.rename(columns={c: f"hsc_{c}" for c in colunas})
    final = final.rename(columns={"specz_redshift": "redshift"})
    SAIDA.parent.mkdir(parents=True, exist_ok=True)
    final.to_csv(SAIDA, index=False)
    print(f"salvo em {SAIDA} ({len(final)} galáxias)")


if __name__ == "__main__":
    main()