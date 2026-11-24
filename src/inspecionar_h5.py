"""Mostra o conteúdo dos arquivos HDF5 de imagens (TransferZ-Images e GalaxiesML).

Para cada arquivo, lista as chaves, o formato, o tipo e alguns valores.
Não carrega as imagens inteiras na memória.
"""
from pathlib import Path

import h5py
import numpy as np

PASTA = Path("data/raw/imagens")
ARQUIVOS = sorted(PASTA.glob("transferz/*.h5")) + sorted(PASTA.glob("galaxiesml/*.hdf5"))


def mostrar(nome, item):
    if not isinstance(item, h5py.Dataset):
        print(f"  [grupo] {nome}")
        return
    print(f"  {nome:40s} formato={item.shape}  tipo={item.dtype}")
    if item.ndim == 1 and len(item) > 0:
        print(f"      primeiros valores: {item[:3]}")
        if np.issubdtype(item.dtype, np.number):
            amostra = item[: min(len(item), 20000)]
            print(f"      nos primeiros {len(amostra)}: min={amostra.min()}  max={amostra.max()}")
    elif item.ndim == 4:
        bloco = item[:64].astype(np.float64)
        print(f"      64 primeiras imagens: min={bloco.min():.4g}  max={bloco.max():.4g}  "
              f"media={bloco.mean():.4g}  NaN={int(np.isnan(bloco).sum())}")


for arquivo in ARQUIVOS:
    print(f"\n=== {arquivo.name} ===")
    with h5py.File(arquivo, "r") as f:
        f.visititems(mostrar)