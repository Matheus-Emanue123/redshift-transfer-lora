"""Cruzamento do conjunto A (fotométrico, equivalente ao TransferZ).

Passo 1: junta COSMOS2020 Classic e Farmer pelo identificador
         (Classic.id_farmer = Farmer.id). Fica só quem existe nas duas.
         Referência do TransferZ: 923.403 objetos.
Passo 2: cruza com o HSC pela posição. Para cada objeto do COSMOS2020,
         procura o objeto HSC mais próximo; mantém o par se a distância
         for menor que 1 segundo de arco. Se dois objetos do COSMOS2020
         apontarem para o mesmo objeto HSC, fica só o par mais próximo.
         Antes do cruzamento, usamos só objetos HSC com isprimary = True
         (remove duplicatas do HSC).

Entradas:
  data/raw/cosmos2020/classic.csv
  data/raw/cosmos2020/farmer.csv
  data/raw/hsc_pdr2_cosmos_median.csv.gz
Saída:
  data/interim/conjunto_a_cruzado.csv

Uso (da raiz do repositório):
  python src/cruzamento_a.py
"""
from pathlib import Path

import astropy.units as u
import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord

CLASSIC = Path("data/raw/cosmos2020/classic.csv")
FARMER = Path("data/raw/cosmos2020/farmer.csv")
HSC = Path("data/raw/hsc_pdr2_wide_cosmos.csv.gz")
SAIDA = Path("data/interim/conjunto_a_cruzado.csv")
RAIO_ARCSEC = 1.0


def ler_hsc(caminho):
    """Lê o arquivo do HSC. O servidor pode incluir linhas de metadados
    no início do arquivo; procuramos a linha de cabeçalho com 'object_id'."""
    import gzip
    abrir = gzip.open if str(caminho).endswith(".gz") else open
    with abrir(caminho, "rt", encoding="utf-8") as f:
        for n, linha in enumerate(f):
            if "object_id" in linha:
                break
        else:
            raise ValueError("cabeçalho com 'object_id' não encontrado no arquivo do HSC")
    df = pd.read_csv(caminho, skiprows=n)
    df.columns = [c.lstrip("# ").strip() for c in df.columns]
    return df


def main():
    # ---------- Passo 1: Classic + Farmer ----------
    classic = pd.read_csv(CLASSIC)
    farmer = pd.read_csv(FARMER)
    print(f"Classic: {len(classic)} objetos | Farmer: {len(farmer)} objetos")

    # colunas da Farmer recebem o sufixo _f
    farmer = farmer.add_suffix("_f")
    cosmos = classic.merge(farmer, left_on="id_farmer", right_on="id_f", how="inner")
    print(f"Passo 1, em comum (Classic.id_farmer = Farmer.id): {len(cosmos)} "
          f"(TransferZ: 923.403)")

    # conferência: o identificador inverso também deve bater
    bate = (cosmos["id_classic_f"] == cosmos["id"]).mean() * 100
    print(f"  conferência Farmer.id_classic = Classic.id: {bate:.2f}% das linhas")

    # ---------- Passo 2: cruzamento com o HSC ----------
    hsc = ler_hsc(HSC)
    print(f"HSC: {len(hsc)} objetos")
    prim = hsc["isprimary"].astype(str).str.lower().isin(["t", "true", "1"])
    hsc = hsc[prim].reset_index(drop=True)
    print(f"HSC com isprimary = True: {len(hsc)}")

    c_cosmos = SkyCoord(ra=cosmos["ra"].values * u.deg, dec=cosmos["dec"].values * u.deg)
    c_hsc = SkyCoord(ra=hsc["ra"].values * u.deg, dec=hsc["dec"].values * u.deg)
    idx, sep, _ = c_cosmos.match_to_catalog_sky(c_hsc)
    sep_arcsec = sep.to(u.arcsec).value

    pares = pd.DataFrame({"i_cosmos": np.arange(len(cosmos)), "i_hsc": idx, "sep_arcsec": sep_arcsec})
    pares = pares[pares["sep_arcsec"] < RAIO_ARCSEC]
    # um objeto HSC só pode ficar com um objeto COSMOS (o mais próximo)
    pares = pares.sort_values("sep_arcsec").drop_duplicates("i_hsc", keep="first")

    hsc_sel = hsc.iloc[pares["i_hsc"].values].reset_index(drop=True).add_prefix("hsc_")
    cos_sel = cosmos.iloc[pares["i_cosmos"].values].reset_index(drop=True)
    sep_col = pd.DataFrame({"sep_arcsec": pares["sep_arcsec"].values})
    final = pd.concat([cos_sel, hsc_sel, sep_col], axis=1)
    print(f"Passo 2, pares a menos de {RAIO_ARCSEC}\": {len(final)}")
    print(f"  distância mediana dos pares: {final['sep_arcsec'].median():.3f}\"")

    SAIDA.parent.mkdir(parents=True, exist_ok=True)
    final.to_csv(SAIDA, index=False)
    print(f"salvo em {SAIDA}")


if __name__ == "__main__":
    main()
