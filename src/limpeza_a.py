"""Limpeza do conjunto A, seguindo o apêndice A.3 do TransferZ.

Aplica os critérios em ordem e mostra quantas galáxias sobram após cada um.
Colunas sem sufixo: COSMOS2020 Classic. Sufixo _f: Farmer. Prefixo hsc_: HSC PDR2.

Decisões nossas (o artigo não define os valores):
  - lp_type "disponível" = valor 0, 1 ou 2 (o -9, falha no ajuste, conta como
    indisponível e usa a classificação do HSC).
  - 23 filtros válidos exigidos na Classic E na Farmer (EXIGIR_FARMER_23).
  - |lp_zpdf - lp_zminchi2| < 0,1 aplicado ao LePhare da Classic e da Farmer.
  - Concordância entre configurações: |lp_zpdf - outro redshift| < 0,1.
  - Valores negativos de redshift ou de chi2 são tratados como ausentes.

Entrada: data/interim/conjunto_a_cruzado.csv
Saída:   data/processed/conjunto_a.csv

Uso (da raiz do repositório):
  python src/limpeza_a.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

ENTRADA = Path("data/interim/conjunto_a_cruzado.csv")
SAIDA = Path("data/processed/conjunto_a.csv")

MARGEM_CONCORDANCIA = 0.1
EXIGIR_FARMER_23 = True

HSC_B = ["g", "r", "i", "z", "y"]
SC = ["ib427", "ib464", "ia484", "ib505", "ia527", "ib574",
      "ia624", "ia679", "ib709", "ia738", "ia767", "ib827"]
UV = ["y", "j", "h", "ks"]
FILTROS_23_CLASSIC = ([f"hsc_{b}_mag_aper2" for b in HSC_B] + [f"sc_{b}_mag_aper2" for b in SC]
                      + [f"uvista_{b}_mag_aper2" for b in UV] + ["irac_ch1_mag", "irac_ch2_mag"])
FILTROS_23_FARMER = ([f"hsc_{b}_mag_f" for b in HSC_B] + [f"sc_{b}_mag_f" for b in SC]
                     + [f"uvista_{b}_mag_f" for b in UV] + ["irac_ch1_mag_f", "irac_ch2_mag_f"])
CMODEL_HSC = [f"hsc_{b}_cmodel_mag" for b in HSC_B]

# as 4 configurações de redshift e o chi2 de cada uma
Z = {"classic_lephare": "lp_zpdf", "classic_eazy": "ez_z_phot",
     "farmer_lephare": "lp_zpdf_f", "farmer_eazy": "ez_z_phot_f"}
CHI2 = {"classic_lephare": "lp_chi2_best", "classic_eazy": "ez_z_phot_chi2",
        "farmer_lephare": "lp_chi2_best_f", "farmer_eazy": "ez_z_phot_chi2_f"}


def entre_0_e_50(df, colunas):
    v = df[colunas]
    return ((v > 0) & (v < 50)).all(axis=1)


def main():
    df = pd.read_csv(ENTRADA)
    passos = []

    def registrar(nome, mascara):
        nonlocal df
        df = df[mascara].copy()
        passos.append((nome, len(df)))
        print(f"{nome:<55} {len(df):>9}")

    print(f"{'entrada (após o cruzamento)':<55} {len(df):>9}")

    # 3. só galáxias
    tipo_ok = df["lp_type"].isin([0, 1, 2])
    galaxia = np.where(tipo_ok, df["lp_type"] == 0, df["hsc_i_extendedness_value"] == 1)
    registrar("3. galáxias (lp_type=0 ou extendedness=1)", galaxia)
    print(f"{'   (TransferZ após passos 2 e 3: 670.053)':<55}")

    # 4. 23 filtros entre 0 e 50 mag
    ok23 = entre_0_e_50(df, FILTROS_23_CLASSIC)
    if EXIGIR_FARMER_23:
        ok23 &= entre_0_e_50(df, FILTROS_23_FARMER)
    registrar("4. 23 filtros entre 0 e 50 mag", ok23)

    # 5. redshift nas 4 configurações
    tem_z = pd.concat([df[c].notna() & (df[c] >= 0) for c in Z.values()], axis=1).all(axis=1)
    registrar("5. redshift nas 4 configurações", tem_z)

    # 6 e 7. chi2: Classic/LePhare < 1; nas outras, limite que remove a mesma fração
    chi_validos = pd.concat([df[c].notna() & (df[c] >= 0) for c in CHI2.values()], axis=1).all(axis=1)
    registrar("6/7a. chi2 válido nas 4 configurações", chi_validos)
    ref = df[CHI2["classic_lephare"]] < 1
    fracao_mantida = ref.mean()
    mascara_chi = ref.copy()
    print(f"   chi2 < 1 (Classic/LePhare) mantém {fracao_mantida * 100:.1f}%")
    for nome in ["classic_eazy", "farmer_lephare", "farmer_eazy"]:
        col = CHI2[nome]
        limite = df[col].quantile(fracao_mantida)
        print(f"   limite de chi2 para {nome}: {limite:.3f}")
        mascara_chi &= df[col] <= limite
    registrar("6/7b. chi2 dentro dos limites nas 4 configurações", mascara_chi)

    # 8. |lp_zpdf - lp_zminchi2| < 0,1 (Classic e Farmer)
    lp_ok = ((df["lp_zpdf"] - df["lp_zminchi2"]).abs() < 0.1) & \
            ((df["lp_zpdf_f"] - df["lp_zminchi2_f"]).abs() < 0.1)
    registrar("8. |lp_zpdf - lp_zminchi2| < 0,1", lp_ok)

    # 9. concordância entre as configurações
    conc = pd.concat([(df["lp_zpdf"] - df[Z[n]]).abs() < MARGEM_CONCORDANCIA
                      for n in ["classic_eazy", "farmer_lephare", "farmer_eazy"]], axis=1).all(axis=1)
    registrar(f"9. 4 redshifts concordam (margem {MARGEM_CONCORDANCIA})", conc)

    # 10. redshift < 4
    registrar("10. redshift < 4", df["lp_zpdf"] < 4)

    # 11. cModel do HSC entre 0 e 50 nos 5 filtros
    registrar("11. cModel HSC entre 0 e 50 mag", entre_0_e_50(df, CMODEL_HSC))
    print(f"{'   (TransferZ final: 116.335)':<55}")

    # tabela final: entrada do modelo + rótulo + identificadores
    colunas = (["hsc_object_id", "hsc_ra", "hsc_dec"] + CMODEL_HSC
               + [f"hsc_{b}_cmodel_magsigma" for b in HSC_B]
               + [f"hsc_a_{b}" for b in HSC_B]
               + ["id", "lp_zpdf", "sep_arcsec"])
    final = df[colunas].rename(columns={"id": "cosmos_id", "lp_zpdf": "redshift"})
    SAIDA.parent.mkdir(parents=True, exist_ok=True)
    final.to_csv(SAIDA, index=False)
    print(f"salvo em {SAIDA}")


if __name__ == "__main__":
    main()
