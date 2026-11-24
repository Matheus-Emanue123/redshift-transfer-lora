"""Direcao do ajuste (A -> B contra B -> A) e erro por faixa de redshift, na MLP.

Modelos (3 sementes cada, 100% dos dados):
  so_A      base_s*.pt       treinado so no A (fotometrico)
  so_B      base_inv_s*.pt   treinado so no B (espectroscopico)
  tl_AB     tl_s*.pt         parte do so_A e ajusta no B   (direcao normal)
  lora_AB   lora_s*.pt
  tl_BA     tl_inv_s*.pt     parte do so_B e ajusta no A   (direcao inversa)
  lora_BA   lora_inv_s*.pt
  combo     combo_s*.pt      A e B juntos, do zero

Parte 1, ganho da transferencia: no conjunto de destino, o modelo ajustado e
comparado com o modelo treinado so no destino.
  direcao normal:  tl_AB e lora_AB contra so_B, no teste B
  direcao inversa: tl_BA e lora_BA contra so_A, no teste A
Se o ajustado for pior que o treinado so no destino, houve transferencia
negativa. Metricas na faixa 0,3 < z < 1,5, media e desvio das 3 sementes.

Parte 2, erro por faixa de redshift real (todas as galaxias do teste, sem a
faixa 0,3-1,5): z de 0 a 0,5; 0,5 a 1; 1 a 1,5; 1,5 a 2; 2 a 3; 3 a 4.
Tambem mostra quantas galaxias do treino do A e do B caem em cada faixa.
Hipotese testada: o B tem poucas galaxias de z alto, entao um modelo que
comeca no B erraria mais nessas faixas, mesmo depois do ajuste no A.

Saidas:
  resultados/direcao_ganho.csv
  resultados/direcao_faixas.csv
  resultados/fig_faixas.png

Uso (da raiz do repositorio):
  python src/analise_faixas.py
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from avaliar import prever, metricas, estilo, DADOS, TESTES, COR, TINTA, TINTA2, GRADE, FUNDO
from treinar import BANDAS

MODELOS = Path("modelos")
RESULTADOS = Path("resultados")
SEMENTES = [0, 1, 2]
ARQUIVO = {"so_A": "base", "so_B": "base_inv", "tl_AB": "tl", "lora_AB": "lora",
           "tl_BA": "tl_inv", "lora_BA": "lora_inv", "combo": "combo"}
ROTULO = {"so_A": "Só A", "so_B": "Só B", "tl_AB": "TL A→B", "lora_AB": "LoRA A→B",
          "tl_BA": "TL B→A", "lora_BA": "LoRA B→A", "combo": "Combo"}
FAIXAS = [0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0]
NOMES_FAIXAS = ["0–0,5", "0,5–1", "1–1,5", "1,5–2", "2–3", "3–4"]


def disp(dz):
    return 1.4826 * np.median(np.abs(dz - np.median(dz))) if len(dz) else np.nan


def main():
    testes = {}
    for nome, arq in TESTES.items():
        df = pd.read_csv(DADOS / f"{arq}.csv")
        testes[nome] = (df[BANDAS].to_numpy(np.float32), df["redshift"].to_numpy(np.float32))

    geral, faixas = [], []
    for chave, arq in ARQUIVO.items():
        for s in SEMENTES:
            info = torch.load(MODELOS / f"{arq}_s{s}.pt", map_location="cpu", weights_only=False)
            for nome, (x, z) in testes.items():
                zp = prever(info, x)
                geral.append({"modelo": chave, "semente": s, "teste": nome, **metricas(zp, z)})
                dz = (zp - z) / (1 + z)
                idx = np.digitize(z, FAIXAS[1:-1])
                for k in range(len(NOMES_FAIXAS)):
                    sel = idx == k
                    faixas.append({"modelo": chave, "semente": s, "teste": nome,
                                   "faixa": NOMES_FAIXAS[k], "n": int(sel.sum()),
                                   "vies": np.median(dz[sel]) if sel.any() else np.nan,
                                   "dispersao": disp(dz[sel]),
                                   "outliers_pct": 100 * np.mean(np.abs(zp - z)[sel] > 1.0) if sel.any() else np.nan})

    RESULTADOS.mkdir(exist_ok=True)
    g = pd.DataFrame(geral)
    f = pd.DataFrame(faixas)
    gm = g.groupby(["teste", "modelo"])[["vies", "dispersao", "outliers_pct"]].agg(["mean", "std"])
    gm.columns = ["_".join(c) for c in gm.columns]
    gm = gm.reset_index()
    gm.to_csv(RESULTADOS / "direcao_ganho.csv", index=False, float_format="%.5f")
    fm = f.groupby(["teste", "modelo", "faixa"], sort=False)[["n", "vies", "dispersao", "outliers_pct"]].mean().reset_index()
    fm.to_csv(RESULTADOS / "direcao_faixas.csv", index=False, float_format="%.5f")

    def linha(teste, modelo):
        r = gm[(gm["teste"] == teste) & (gm["modelo"] == modelo)].iloc[0]
        return (f"{ROTULO[modelo]:<10}{r['vies_mean']:>9.4f} ±{r['vies_std']:.4f}"
                f"{r['dispersao_mean']:>9.4f} ±{r['dispersao_std']:.4f}{r['outliers_pct_mean']:>8.2f}")

    print("Parte 1: ganho da transferencia (faixa 0,3 < z < 1,5, media ± desvio de 3 sementes)")
    print(f"{'modelo':<10}{'vies':>17}{'dispersao':>17}{'outl %':>8}")
    print("Direcao normal, teste B (destino):")
    for mdl in ["so_B", "tl_AB", "lora_AB", "so_A", "combo"]:
        print("  " + linha("B", mdl))
    print("Direcao inversa, teste A (destino):")
    for mdl in ["so_A", "tl_BA", "lora_BA", "so_B", "combo"]:
        print("  " + linha("A", mdl))
    print("Teste Combo:")
    for mdl in ARQUIVO:
        print("  " + linha("Combo", mdl))

    print("\nParte 2: galaxias de treino por faixa de z")
    za = pd.read_csv(DADOS / "a_treino.csv")["redshift"].to_numpy()
    zb = pd.read_csv(DADOS / "b_treino.csv")["redshift"].to_numpy()
    ca = np.histogram(za, FAIXAS)[0]
    cb = np.histogram(zb, FAIXAS)[0]
    print(f"{'faixa':<8}{'A':>9}{'A %':>7}{'B':>9}{'B %':>7}")
    for k, nf in enumerate(NOMES_FAIXAS):
        print(f"{nf:<8}{ca[k]:>9}{100 * ca[k] / ca.sum():>7.1f}{cb[k]:>9}{100 * cb[k] / cb.sum():>7.1f}")

    print("\nParte 2: dispersao por faixa de z (media de 3 sementes)")
    for teste in ["A", "B"]:
        print(f"Teste {teste}")
        print(f"  {'modelo':<10}" + "".join(f"{nf:>9}" for nf in NOMES_FAIXAS))
        for mdl in ["so_A", "so_B", "tl_AB", "lora_AB", "tl_BA", "lora_BA", "combo"]:
            d = fm[(fm["teste"] == teste) & (fm["modelo"] == mdl)].set_index("faixa").reindex(NOMES_FAIXAS)
            print(f"  {ROTULO[mdl]:<10}" + "".join(f"{v:>9.4f}" for v in d["dispersao"]))
        n = fm[(fm["teste"] == teste) & (fm["modelo"] == "so_A")].set_index("faixa").reindex(NOMES_FAIXAS)["n"]
        print(f"  {'galaxias':<10}" + "".join(f"{int(v):>9}" for v in n))

    figura(fm, ca, cb)
    print(f"\nsalvo em {RESULTADOS}/: direcao_ganho.csv, direcao_faixas.csv, fig_faixas.png")


def figura(fm, ca, cb):
    estilos = {"so_A": (COR["base"], "--"), "so_B": (COR["combo"], "--"),
               "tl_AB": (COR["tl"], "-"), "lora_AB": (COR["lora"], "-"),
               "tl_BA": (COR["tl"], ":"), "lora_BA": (COR["lora"], ":")}
    fig, eixos = plt.subplots(1, 3, figsize=(13, 4), facecolor=FUNDO)
    pos = np.arange(len(NOMES_FAIXAS))
    ax = eixos[0]
    estilo(ax)
    larg = 0.38
    ax.bar(pos - larg / 2, 100 * ca / ca.sum(), larg, color=COR["base"], label="Treino do A")
    ax.bar(pos + larg / 2, 100 * cb / cb.sum(), larg, color=COR["combo"], label="Treino do B")
    ax.set_xticks(pos, NOMES_FAIXAS, fontsize=8)
    ax.set_ylabel("% das galáxias de treino", color=TINTA, fontsize=10)
    ax.set_title("Distribuição de z no treino", color=TINTA, fontsize=11)
    ax.legend(frameon=False, fontsize=9, labelcolor=TINTA)
    for ax, teste in zip(eixos[1:], ["A", "B"]):
        estilo(ax)
        for mdl, (cor, traco) in estilos.items():
            d = fm[(fm["teste"] == teste) & (fm["modelo"] == mdl)].set_index("faixa").reindex(NOMES_FAIXAS)
            ax.plot(pos, d["dispersao"], color=cor, linestyle=traco, linewidth=2, marker="o",
                    markersize=6, markeredgecolor=FUNDO, label=ROTULO[mdl])
        ax.set_xticks(pos, NOMES_FAIXAS, fontsize=8)
        ax.set_title(f"Teste {teste}", color=TINTA, fontsize=11)
        ax.set_xlabel("z real", color=TINTA2, fontsize=9)
        ax.set_yscale("log")
    eixos[1].set_ylabel("Dispersão de dz (log)", color=TINTA, fontsize=10)
    alcas, rotulos = eixos[1].get_legend_handles_labels()
    fig.legend(alcas, rotulos, loc="upper center", ncol=6, frameon=False, fontsize=9, labelcolor=TINTA)
    fig.text(0.5, 0.01, "Tracejado: treino num conjunto só. Contínuo: ajuste A→B. Pontilhado: ajuste B→A.",
             ha="center", color=TINTA2, fontsize=8.5)
    fig.tight_layout(rect=(0, 0.04, 1, 0.9))
    fig.savefig(RESULTADOS / "fig_faixas.png", dpi=200, facecolor=FUNDO)
    plt.close(fig)


if __name__ == "__main__":
    main()