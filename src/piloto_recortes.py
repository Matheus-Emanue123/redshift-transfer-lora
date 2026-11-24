"""Piloto de download de recortes em lote: 1.000 galáxias do conjunto A ou B.

Objetivo: medir o tempo de download e o tamanho dos arquivos, para decidir
quantas galáxias usar. Segue o manual do DAS Cutout:
  - envio de uma lista de coordenadas (multipart/form-data, campo "list")
  - primeira linha da lista: "#? ..." com os nomes das colunas
  - lista ordenada por filtro e depois por posição (mais rápido no servidor)
  - resposta: um arquivo .tar sem compressão com os FITS
  - limite de 100.000 imagens por pedido (aqui são 5.000)

Resultado do teste_recorte.py: endereço do PDR2 funciona com rerun pdr2_wide;
pixel de 0,168". Semi-lado de 32 x 0,168 = 5,376" para recortes de 64 x 64.

Resultado do piloto do conjunto B: 5.000 de 5.000 imagens, 30,9 KB por imagem,
105,7 min para 1.000 galáxias sorteadas no levantamento Wide inteiro.

Cada conjunto usa um rerun (nome da pasta de imagens no HSC):
  A: pdr2_dud_wide_depth_median  (COSMOS wide-depth mediano, ainda não testado)
  B: pdr2_wide

Saídas (X = a ou b):
  data/raw/piloto_recortes_X/lista.txt
  data/raw/piloto_recortes_X/recortes.tar
  data/raw/piloto_recortes_X/fits/   (arquivos extraídos)

Uso (da raiz do repositório, com as variáveis de usuário e senha do HSC):
  python src/piloto_recortes.py a
  python src/piloto_recortes.py b
"""
import os
import sys
import tarfile
import time
from pathlib import Path

import pandas as pd
import requests

ENDERECO = "https://hsc-release.mtk.nao.ac.jp/das_cutout/pdr2/cgi-bin/cutout"
CONJUNTOS = {
    "a": {"arquivo": "data/processed/conjunto_a.csv", "rerun": "pdr2_wide"},
    "b": {"arquivo": "data/processed/conjunto_b.csv", "rerun": "pdr2_wide"},
}
FILTROS = ["HSC-G", "HSC-R", "HSC-I", "HSC-Z", "HSC-Y"]
SEMI_LADO = "5.376asec"
N_GALAXIAS = 1000
SEMENTE = 42
TOTAL_B = 285_578
TOTAL_A = 96_473


def main(conjunto):
    rerun = CONJUNTOS[conjunto]["rerun"]
    PASTA = Path(f"data/raw/piloto_recortes_{conjunto}")
    usuario = os.environ.get("HSC_SSP_CAS_USER")
    senha = os.environ.get("HSC_SSP_CAS_PASSWORD")
    if not usuario or not senha:
        print("defina HSC_SSP_CAS_USER e HSC_SSP_CAS_PASSWORD no terminal antes de rodar")
        sys.exit(1)

    tabela = pd.read_csv(CONJUNTOS[conjunto]["arquivo"],
                         usecols=["hsc_object_id", "hsc_ra", "hsc_dec"])
    amostra = tabela.sample(n=N_GALAXIAS, random_state=SEMENTE)
    print(f"conjunto {conjunto.upper()}, rerun {rerun}")

    # teste rápido: 1 imagem com este rerun, antes de enviar o lote
    # (no lote, o servidor ignora erros sem avisar e só entregaria um .tar vazio)
    g0 = amostra.iloc[0]
    params = {"ra": g0["hsc_ra"], "dec": g0["hsc_dec"], "sw": SEMI_LADO, "sh": SEMI_LADO,
              "type": "coadd", "image": "on", "filter": "HSC-G", "tract": "", "rerun": rerun}
    r = requests.get(ENDERECO, params=params, auth=(usuario, senha), timeout=120)
    if r.status_code != 200 or r.content[:6] != b"SIMPLE":
        print(f"teste de 1 imagem falhou (HTTP {r.status_code}): {r.text[:300]}")
        sys.exit(1)
    print("teste de 1 imagem: ok")

    # uma linha por (filtro, galáxia); ordenada por filtro, ra, dec (recomendação do manual)
    linhas = []
    for filtro in FILTROS:
        for _, g in amostra.sort_values(["hsc_ra", "hsc_dec"]).iterrows():
            nome = f"{int(g['hsc_object_id'])}_{filtro}"
            linhas.append(f"{rerun} {filtro} {g['hsc_ra']:.7f} {g['hsc_dec']:.7f} "
                          f"{SEMI_LADO} {SEMI_LADO} {nome}")
    lista = "#? rerun filter ra dec sw sh name\n" + "\n".join(linhas) + "\n"

    PASTA.mkdir(parents=True, exist_ok=True)
    (PASTA / "lista.txt").write_text(lista, encoding="ascii")
    print(f"lista com {len(linhas)} imagens ({N_GALAXIAS} galáxias x 5 filtros)")

    # envia a lista e grava a resposta (.tar) direto no disco
    arq_tar = PASTA / "recortes.tar"
    inicio = time.time()
    with requests.post(ENDERECO, files={"list": ("lista.txt", lista)},
                       auth=(usuario, senha), stream=True, timeout=3600) as r:
        print(f"HTTP {r.status_code}, tipo: {r.headers.get('Content-Type')}")
        if r.status_code != 200:
            print("resposta do servidor (início):", r.text[:500])
            sys.exit(1)
        with open(arq_tar, "wb") as f:
            for bloco in r.iter_content(chunk_size=1 << 20):
                f.write(bloco)
    segundos = time.time() - inicio
    mb = arq_tar.stat().st_size / 1e6
    print(f"download: {mb:.1f} MB em {segundos / 60:.1f} min")

    # extrai e confere quantas imagens vieram
    pasta_fits = PASTA / "fits"
    pasta_fits.mkdir(exist_ok=True)
    with tarfile.open(arq_tar) as t:
        membros = [m for m in t.getmembers() if m.isfile() and m.name.endswith(".fits")]
        t.extractall(pasta_fits, members=membros)
    print(f"imagens recebidas: {len(membros)} de {len(linhas)}")
    if membros:
        print("exemplos de nomes:", [m.name for m in membros[:3]])
        kb = sum(m.size for m in membros) / len(membros) / 1024
        print(f"tamanho médio por imagem: {kb:.1f} KB")

        # estimativas para os conjuntos inteiros
        por_galaxia_s = segundos / N_GALAXIAS
        por_galaxia_mb = mb / N_GALAXIAS
        for nome, total in [("B", TOTAL_B), ("A", TOTAL_A)]:
            print(f"estimativa conjunto {nome} ({total} galáxias): "
                  f"{total * por_galaxia_mb / 1000:.1f} GB, {total * por_galaxia_s / 3600:.1f} h")


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in CONJUNTOS:
        print("uso: python src/piloto_recortes.py a|b")
        sys.exit(1)
    main(sys.argv[1])