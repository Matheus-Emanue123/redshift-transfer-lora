"""Avaliação de todos os modelos treinados, no conjunto de teste.

Para cada modelo em modelos/*.pt, calcula as métricas em 3 conjuntos de teste:
a_teste (fotométrico), b_teste (espectroscópico) e combo_teste (os dois).

Métricas (artigo da LoRA), só para galáxias com 0,3 < z_real < 1,5:
  dz          = (z_previsto - z_real) / (1 + z_real)
  viés        = mediana(dz)
  dispersão   = 1,4826 x mediana(|dz - mediana(dz)|)     (desvio absoluto mediano)
  disp_iqr    = (percentil 75 - percentil 25 de dz) / 1,349   (definição do TransferZ)
  outliers    = % de galáxias com |z_previsto - z_real| > 1,0
Limites do LSST: |viés| < 0,003 e dispersão < 0,02.

Saídas:
  resultados/metricas.csv   uma linha por modelo e conjunto de teste
  resultados/resumo.csv     média e desvio padrão entre as sementes
  resultados/fig_fracoes.png       métricas por fração do treino do B (opção A)
  resultados/fig_previsto_real.png z previsto contra z real (modelos com 100%, semente 0)

Uso (da raiz do repositório):
  python src/avaliar.py
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from treinar import MLP, BANDAS   # mesma arquitetura do treino

DADOS = Path("data/modelo")
MODELOS = Path("modelos")
RESULTADOS = Path("resultados")
TESTES = {"A": "a_teste", "B": "b_teste", "Combo": "combo_teste"}
Z_MIN, Z_MAX = 0.3, 1.5
LSST_VIES, LSST_DISP = 0.003, 0.02

# cores: 3 primeiras posições da paleta categórica de referência (validadas em conjunto)
COR = {"tl": "#2a78d6", "lora": "#eb6834", "combo": "#1baf7a", "base": "#52514e"}
NOME = {"base": "MLP-Base", "tl": "MLP-TL", "lora": "MLP-LoRA", "combo": "MLP-Combo"}
TINTA, TINTA2, GRADE, FUNDO = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"


def metricas(z_prev, z_real):
    faixa = (z_real > Z_MIN) & (z_real < Z_MAX)
    zp, zr = z_prev[faixa], z_real[faixa]
    dz = (zp - zr) / (1 + zr)
    med = np.median(dz)
    q75, q25 = np.percentile(dz, [75, 25])
    return {
        "n_faixa": int(faixa.sum()),
        "vies": med,
        "dispersao": 1.4826 * np.median(np.abs(dz - med)),
        "disp_iqr": (q75 - q25) / 1.349,
        "outliers_pct": 100 * np.mean(np.abs(zp - zr) > 1.0),
    }


def prever(info, x):
    modelo = MLP()
    modelo.load_state_dict(info["estado"])
    modelo.eval()
    xn = torch.tensor((x - info["media"]) / info["desvio"], dtype=torch.float32)
    with torch.no_grad():
        return modelo(xn).numpy()


def main():
    testes = {}
    for nome, arq in TESTES.items():
        df = pd.read_csv(DADOS / f"{arq}.csv")
        testes[nome] = (df[BANDAS].to_numpy(np.float32), df["redshift"].to_numpy(np.float32))

    linhas, previsoes = [], {}
    arquivos = sorted(MODELOS.glob("*.pt"))
    print(f"{len(arquivos)} modelos encontrados")
    for arq in arquivos:
        info = torch.load(arq, map_location="cpu", weights_only=False)
        for nome, (x, z) in testes.items():
            zp = prever(info, x)
            linhas.append({"estrategia": info["estrategia"], "fracao": info["fracao"],
                           "semente": info["semente"], "teste": nome, **metricas(zp, z)})
            if info["fracao"] == 100 and info["semente"] == 0 and nome == "Combo":
                previsoes[info["estrategia"]] = (zp, z)

    RESULTADOS.mkdir(exist_ok=True)
    m = pd.DataFrame(linhas).sort_values(["teste", "estrategia", "fracao", "semente"])
    m.to_csv(RESULTADOS / "metricas.csv", index=False, float_format="%.5f")

    cols = ["vies", "dispersao", "disp_iqr", "outliers_pct"]
    resumo = m.groupby(["teste", "estrategia", "fracao"])[cols].agg(["mean", "std"])
    resumo.columns = ["_".join(c) for c in resumo.columns]
    resumo["n_sementes"] = m.groupby(["teste", "estrategia", "fracao"]).size()
    resumo = resumo.reset_index()
    resumo.to_csv(RESULTADOS / "resumo.csv", index=False, float_format="%.5f")

    # tabela principal no terminal: modelos com 100% do B
    print(f"\nModelos com 100% do treino do B (média entre sementes), faixa {Z_MIN} < z < {Z_MAX}")
    print(f"{'teste':<6}{'modelo':<11}{'viés':>10}{'dispersão':>11}{'outliers %':>12}{'sementes':>10}")
    for _, r in resumo[resumo["fracao"] == 100].iterrows():
        print(f"{r['teste']:<6}{NOME[r['estrategia']]:<11}{r['vies_mean']:>10.4f}"
              f"{r['dispersao_mean']:>11.4f}{r['outliers_pct_mean']:>12.2f}{int(r['n_sementes']):>10}")
    print(f"limites do LSST: |viés| < {LSST_VIES}, dispersão < {LSST_DISP}")

    figura_fracoes(resumo)
    figura_previsto_real(previsoes)
    print(f"\nsalvo em {RESULTADOS}/: metricas.csv, resumo.csv, fig_fracoes.png, fig_previsto_real.png")


def estilo(ax):
    ax.set_facecolor(FUNDO)
    for lado in ("top", "right"):
        ax.spines[lado].set_visible(False)
    for lado in ("left", "bottom"):
        ax.spines[lado].set_color(GRADE)
    ax.tick_params(colors=TINTA2, labelsize=9)
    ax.grid(axis="y", color=GRADE, linewidth=0.8)
    ax.set_axisbelow(True)


def figura_fracoes(resumo):
    """Pequenos múltiplos: linhas = métrica, colunas = conjunto de teste."""
    metr = [("dispersao", "Dispersão", LSST_DISP), ("vies", "Viés", LSST_VIES),
            ("outliers_pct", "Outliers (%)", None)]
    fracoes = [10, 25, 50, 100]
    pos = np.arange(len(fracoes))
    fig, eixos = plt.subplots(len(metr), len(TESTES), figsize=(11, 8.5), facecolor=FUNDO)
    for j, teste in enumerate(TESTES):
        for i, (col, rotulo, limite) in enumerate(metr):
            ax = eixos[i, j]
            estilo(ax)
            r = resumo[resumo["teste"] == teste]
            base = r[r["estrategia"] == "base"]
            if len(base):
                ax.axhline(base[f"{col}_mean"].iloc[0], color=COR["base"], linewidth=2,
                           linestyle="--", label=NOME["base"])
            for est in ("tl", "lora", "combo"):
                d = r[r["estrategia"] == est].set_index("fracao").reindex(fracoes)
                if d[f"{col}_mean"].notna().any():
                    ax.errorbar(pos, d[f"{col}_mean"], yerr=d[f"{col}_std"].fillna(0),
                                color=COR[est], linewidth=2, marker="o", markersize=7,
                                markeredgecolor=FUNDO, markeredgewidth=1.5, capsize=3,
                                label=NOME[est])
            if limite is not None:
                ax.axhline(limite, color=TINTA2, linewidth=1, linestyle=":")
                if col == "vies":
                    ax.axhline(-limite, color=TINTA2, linewidth=1, linestyle=":")
                    ax.axhline(0, color=GRADE, linewidth=1)
            ax.set_xticks(pos, [f"{f}%" for f in fracoes])
            if i == 0:
                ax.set_title(f"Teste {teste}", color=TINTA, fontsize=11)
            if j == 0:
                ax.set_ylabel(rotulo, color=TINTA, fontsize=10)
            if i == len(metr) - 1:
                ax.set_xlabel("Fração do treino do B", color=TINTA2, fontsize=9)
    alcas, rotulos = eixos[0, 0].get_legend_handles_labels()
    fig.legend(alcas, rotulos, loc="upper center", ncol=4, frameon=False, fontsize=10,
               labelcolor=TINTA)
    fig.text(0.5, 0.005, f"Linhas pontilhadas: limites do LSST (dispersão {LSST_DISP}, viés ±{LSST_VIES}). "
             f"Barras: desvio padrão entre sementes. Faixa {Z_MIN} < z < {Z_MAX}.",
             ha="center", color=TINTA2, fontsize=8.5)
    fig.tight_layout(rect=(0, 0.02, 1, 0.95))
    fig.savefig(RESULTADOS / "fig_fracoes.png", dpi=200, facecolor=FUNDO)
    plt.close(fig)


def figura_previsto_real(previsoes):
    """z previsto contra z real no teste Combo, modelos com 100% e semente 0."""
    ordem = [e for e in ("base", "tl", "lora", "combo") if e in previsoes]
    if not ordem:
        return
    fig, eixos = plt.subplots(1, len(ordem), figsize=(3.3 * len(ordem), 3.6), facecolor=FUNDO,
                              squeeze=False)
    for ax, est in zip(eixos[0], ordem):
        estilo(ax)
        ax.grid(False)
        zp, z = previsoes[est]
        ax.hexbin(z, zp, gridsize=60, extent=(0, 4, 0, 4), bins="log", cmap="Blues", mincnt=1)
        ax.plot([0, 4], [0, 4], color=TINTA2, linewidth=1)
        ax.set_xlim(0, 4)
        ax.set_ylim(0, 4)
        ax.set_aspect("equal")
        ax.set_title(NOME[est], color=TINTA, fontsize=11)
        ax.set_xlabel("z real", color=TINTA2, fontsize=9)
    eixos[0, 0].set_ylabel("z previsto", color=TINTA2, fontsize=9)
    fig.tight_layout()
    fig.savefig(RESULTADOS / "fig_previsto_real.png", dpi=200, facecolor=FUNDO)
    plt.close(fig)


if __name__ == "__main__":
    main()