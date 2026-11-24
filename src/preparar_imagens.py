"""Prepara as imagens para a CNN.

Le os arquivos HDF5 do TransferZ-Images (tz) e do GalaxiesML (gm), sorteia uma
parte das galaxias de treino e de validacao e grava tudo em arquivos .npy.
O teste e gravado inteiro.

Motivo: os arquivos HDF5 tem 43,6 GB e nao cabem na memoria. Os .npy menores
podem ser lidos direto do disco durante o treino, sem carregar tudo na memoria.
As imagens sao gravadas em float16 para ocupar metade do espaco.
"""
from pathlib import Path

import h5py
import numpy as np

ENTRADA = Path("data/raw/imagens")
SAIDA = Path("data/imagens")

ARQUIVOS = {
    "tz": {
        "treino": "transferz/transferz_5x64x64_TRAINING.h5",
        "val": "transferz/transferz_5x64x64_VALIDATION.h5",
        "teste": "transferz/transferz_5x64x64_TESTING.h5",
    },
    "gm": {
        "treino": "galaxiesml/5x64x64_training_with_morphology.hdf5",
        "val": "galaxiesml/5x64x64_validation_with_morphology.hdf5",
        "teste": "galaxiesml/5x64x64_testing_with_morphology.hdf5",
    },
}
ROTULO = {"tz": "lp_zPDF_CLASSIC", "gm": "specz_redshift"}

# Quantas galaxias sortear de cada parte. None = todas.
QUANTIDADE = {
    ("tz", "treino"): None,
    ("gm", "treino"): None,
    ("tz", "val"): None,
    ("gm", "val"): None,
    ("tz", "teste"): None,
    ("gm", "teste"): None,
}
SEMENTE = 42
BLOCO = 2048  # linhas lidas do HDF5 por vez


def extrair(conjunto, parte, rng):
    caminho = ENTRADA / ARQUIVOS[conjunto][parte]
    with h5py.File(caminho, "r") as f:
        total = f["image"].shape[0]
        n = QUANTIDADE[(conjunto, parte)]
        if n is None or n >= total:
            indices = np.arange(total)
        else:
            indices = np.sort(rng.choice(total, size=n, replace=False))

        ids = f["object_id"][:][indices]
        z = f[ROTULO[conjunto]][:][indices].astype(np.float32)

        destino = SAIDA / f"{conjunto}_{parte}_img.npy"
        imagens = np.lib.format.open_memmap(
            destino, mode="w+", dtype=np.float16, shape=(len(indices), 5, 64, 64)
        )
        pos, maior = 0, 0.0
        for inicio in range(0, total, BLOCO):
            fim = min(inicio + BLOCO, total)
            a, b = np.searchsorted(indices, [inicio, fim])
            if a == b:
                continue
            bloco = f["image"][inicio:fim][indices[a:b] - inicio]
            maior = max(maior, float(np.abs(bloco).max()))
            imagens[pos:pos + (b - a)] = bloco.astype(np.float16)
            pos += b - a
            if (inicio // BLOCO) % 20 == 0:
                print(f"    {fim}/{total} linhas lidas", flush=True)
        imagens.flush()
        nao_finitos = int((~np.isfinite(imagens[: min(len(indices), 2000)])).sum())
        del imagens

    np.save(SAIDA / f"{conjunto}_{parte}_z.npy", z)
    np.save(SAIDA / f"{conjunto}_{parte}_id.npy", ids)
    print(f"  {conjunto} {parte}: {len(indices)} galaxias de {total}, "
          f"z de {z.min():.3f} a {z.max():.3f}, maior |pixel| = {maior:.1f}, "
          f"valores nao finitos nas 2000 primeiras = {nao_finitos}")


def contar_repetidas():
    """Galaxias que estao nos dois conjuntos (o artigo LoRA cita 557)."""
    ids = {}
    for conjunto in ARQUIVOS:
        partes = []
        for parte in ARQUIVOS[conjunto]:
            with h5py.File(ENTRADA / ARQUIVOS[conjunto][parte], "r") as f:
                partes.append(f["object_id"][:])
        ids[conjunto] = np.concatenate(partes)
    repetidas = np.intersect1d(ids["tz"], ids["gm"])
    np.save(SAIDA / "repetidas_id.npy", repetidas)
    print(f"\nGalaxias presentes nos dois conjuntos: {len(repetidas)}")


if __name__ == "__main__":
    SAIDA.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEMENTE)
    for conjunto in ARQUIVOS:
        for parte in ARQUIVOS[conjunto]:
            print(f"\nLendo {conjunto} {parte}...")
            extrair(conjunto, parte, rng)
    contar_repetidas()
    print("\nArquivos gravados em", SAIDA)