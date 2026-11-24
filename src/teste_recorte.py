"""Teste do serviço de recortes do HSC (DAS Cutout) com 1 galáxia do conjunto B.

Objetivo: responder 3 perguntas antes de baixar imagens em lote.
  1. O serviço aceita o rerun pdr2_wide?
  2. Qual endereço funciona: .../das_cutout/pdr2/ ou .../das_cutout/pdr3/?
  3. Qual é o tamanho do pixel e o tamanho real do recorte?

Pedido de uma imagem, conforme o manual do DAS Cutout:
  .../cgi-bin/cutout?ra=..&dec=..&sw=..&sh=..&type=coadd&image=on&filter=HSC-G&tract=&rerun=..
com usuário e senha (as mesmas variáveis do hsc.py).

Saída: data/raw/teste_recortes/<filtro>.fits

Uso (da raiz do repositório):
  python src/teste_recorte.py
"""
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from astropy.io import fits
from astropy.wcs import WCS

ENDERECOS = [
    "https://hsc-release.mtk.nao.ac.jp/das_cutout/pdr2/cgi-bin/cutout",
    "https://hsc-release.mtk.nao.ac.jp/das_cutout/pdr3/cgi-bin/cutout",
]
RERUN = "pdr2_wide"
FILTROS = ["HSC-G", "HSC-R", "HSC-I", "HSC-Z", "HSC-Y"]
SEMI_LADO = "5asec"     # recorte de 10" x 10"; o tamanho em pixels é medido abaixo
PASTA = Path("data/raw/teste_recortes")


def main():
    usuario = os.environ.get("HSC_SSP_CAS_USER")
    senha = os.environ.get("HSC_SSP_CAS_PASSWORD")
    if not usuario or not senha:
        print("defina HSC_SSP_CAS_USER e HSC_SSP_CAS_PASSWORD no terminal antes de rodar")
        sys.exit(1)

    gal = pd.read_csv("data/processed/conjunto_b.csv", nrows=1).iloc[0]
    ra, dec = gal["hsc_ra"], gal["hsc_dec"]
    print(f"galáxia de teste: object_id {int(gal['hsc_object_id'])}, ra {ra:.6f}, dec {dec:.6f}")
    PASTA.mkdir(parents=True, exist_ok=True)

    # 1) descobre qual endereço aceita o rerun pdr2_wide, usando o filtro g
    endereco = None
    for url in ENDERECOS:
        params = {"ra": ra, "dec": dec, "sw": SEMI_LADO, "sh": SEMI_LADO, "type": "coadd",
                  "image": "on", "filter": "HSC-G", "tract": "", "rerun": RERUN}
        r = requests.get(url, params=params, auth=(usuario, senha), timeout=120)
        print(f"\n{url}\n  HTTP {r.status_code}, {len(r.content)} bytes, tipo: {r.headers.get('Content-Type')}")
        if r.status_code == 200 and r.content[:6] == b"SIMPLE":   # todo arquivo FITS começa com SIMPLE
            endereco = url
            break
        print("  resposta do servidor (início):", r.text[:300].replace("\n", " "))
    if endereco is None:
        print("\nnenhum endereço devolveu um FITS: me mande esta saída")
        sys.exit(1)
    print(f"\nendereço que funcionou: {endereco}")

    # 2) baixa os 5 filtros e mostra o tamanho de cada imagem
    for filtro in FILTROS:
        params = {"ra": ra, "dec": dec, "sw": SEMI_LADO, "sh": SEMI_LADO, "type": "coadd",
                  "image": "on", "filter": filtro, "tract": "", "rerun": RERUN}
        r = requests.get(endereco, params=params, auth=(usuario, senha), timeout=120)
        if r.status_code != 200 or r.content[:6] != b"SIMPLE":
            print(f"{filtro}: falhou (HTTP {r.status_code}) {r.text[:200]}")
            continue
        arq = PASTA / f"{filtro}.fits"
        arq.write_bytes(r.content)
        with fits.open(arq) as h:
            img = h[1].data
            escala = np.abs(WCS(h[1].header).proj_plane_pixel_scales()[0].to("arcsec").value)
            print(f"{filtro}: {img.shape[1]} x {img.shape[0]} pixels, pixel = {escala:.4f}\", "
                  f"arquivo = {arq.stat().st_size / 1024:.1f} KB, {len(h)} camadas")

    print("\nPara um recorte de 64 x 64 pixels, o semi-lado deve ser 32 x (tamanho do pixel).")


if __name__ == "__main__":
    main()