"""Raspagem do COSMOS2020 (Classic e Farmer) pelo serviço TAP do IRSA.

O TransferZ usa as duas versões do catálogo:
- Classic: fornece o rótulo (lp_zpdf) e o cruzamento com o HSC.
- Farmer: usado na limpeza, para comparar o redshift das quatro configurações
  (Classic/LePhare, Classic/EAZY, Farmer/LePhare, Farmer/EAZY).
As duas versões são ligadas por identificador:
  Classic.id_farmer = Farmer.id   e   Farmer.id_classic = Classic.id

Uso (da raiz do repositório):
  python src/collection/cosmos2020.py classic
  python src/collection/cosmos2020.py farmer
"""
import sys
import time
from pathlib import Path

import pandas as pd
import pyvo

# 23 filtros exigidos pelo TransferZ: HSC grizy (5), Suprime-Cam bandas
# médias (12), UltraVISTA YJHKs (4) e Spitzer/IRAC canais 1 e 2 (2)
HSC = ["g", "r", "i", "z", "y"]
SC = ["ib427", "ib464", "ia484", "ib505", "ia527", "ib574",
      "ia624", "ia679", "ib709", "ia738", "ia767", "ib827"]
UVISTA = ["y", "j", "h", "ks"]

COMUNS = [
    "id", "ra", "dec",
    "lp_type", "lp_zpdf", "lp_zbest", "lp_zminchi2", "lp_chi2_best",
    "ez_z_phot", "ez_z_phot_chi2",
    "flag_hsc", "flag_supcam", "flag_uvista", "flag_combined",
]

CONFIG = {
    "classic": {
        "tabela": "cosmos2020c",
        "colunas": COMUNS + ["id_farmer"]
        # Classic tem várias aberturas; usamos a de 2" (aper2)
        + [f"hsc_{b}_mag_aper2" for b in HSC]
        + [f"sc_{b}_mag_aper2" for b in SC]
        + [f"uvista_{b}_mag_aper2" for b in UVISTA]
        + ["irac_ch1_mag", "irac_ch2_mag"]
        # magnitudes HSC do COSMOS, só para conferir o cruzamento depois
        + [f"hsc_{b}_mag_auto" for b in HSC],
    },
    "farmer": {
        "tabela": "cosmos2020f",
        "colunas": COMUNS + ["id_classic"]
        + [f"hsc_{b}_mag" for b in HSC]
        + [f"sc_{b}_mag" for b in SC]
        + [f"uvista_{b}_mag" for b in UVISTA]
        + ["irac_ch1_mag", "irac_ch2_mag"],
    },
}

RA_MIN, RA_MAX, PASSO = 148.0, 152.0, 0.05   # faixas menores: mais colunas por linha
TAP = pyvo.dal.TAPService("https://irsa.ipac.caltech.edu/TAP")


def baixar(query, tentativas=4):
    for i in range(tentativas):
        try:
            return TAP.run_async(query, maxrec=500_000).to_table().to_pandas()
        except Exception as e:
            print(f"  falhou ({type(e).__name__}), tentativa {i + 1}/{tentativas}")
            time.sleep(5 * (i + 1))
    raise RuntimeError("não foi possível baixar este pedaço")


def main(versao):
    cfg = CONFIG[versao]
    tabela, colunas = cfg["tabela"], cfg["colunas"]
    pasta = Path(f"data/raw/cosmos2020/{versao}_chunks")
    saida = Path(f"data/raw/cosmos2020/{versao}.csv")
    pasta.mkdir(parents=True, exist_ok=True)

    total = int(TAP.run_sync(f"SELECT COUNT(*) AS n FROM {tabela}").to_table()["n"][0])
    print(f"{tabela}: {total} linhas, {len(colunas)} colunas")

    n_faixas = round((RA_MAX - RA_MIN) / PASSO)
    for k in range(n_faixas):
        lo = RA_MIN + k * PASSO
        hi = lo + PASSO
        arq = pasta / f"ra_{lo:.2f}_{hi:.2f}.csv"
        if arq.exists():            # já baixado, pula (permite retomar)
            continue
        q = (f"SELECT {', '.join(colunas)} FROM {tabela} "
             f"WHERE ra >= {lo:.2f} AND ra < {hi:.2f}")
        df = baixar(q)
        df.to_csv(arq, index=False)
        print(f"{arq.name}: {len(df)} linhas")

    # junta só os pedaços com dados (evita o aviso do pandas sobre arquivos vazios)
    partes = [pd.read_csv(f) for f in sorted(pasta.glob("ra_*.csv"))]
    partes = [p for p in partes if len(p) > 0]
    final = pd.concat(partes, ignore_index=True)
    final.to_csv(saida, index=False)
    print(f"{len(final)} de {total} linhas salvas em {saida}")
    if len(final) != total:
        print("ATENÇÃO: faltam linhas, a faixa de RA pode não cobrir toda a tabela")


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in CONFIG:
        print("uso: python src/collection/cosmos2020.py classic|farmer")
        sys.exit(1)
    main(sys.argv[1])   