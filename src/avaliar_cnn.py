"""Avaliacao das 4 CNNs (base, tl, lora, combo) nos testes completos.

Testes:
  A      teste do TransferZ-Images (sem as 557 galaxias repetidas)
  B      teste do GalaxiesML
  Combo  os dois juntos

Metricas iguais as da MLP (artigo LoRA), so para 0,3 < z_real < 1,5:
  dz        = (z_previsto - z_real) / (1 + z_real)
  vies      = mediana(dz)
  dispersao = 1,4826 x mediana(|dz - mediana(dz)|)
  disp_iqr  = (percentil 75 - percentil 25 de dz) / 1,349
  outliers  = % de galaxias com |z_previsto - z_real| > 1,0

Saidas:
  resultados/cnn_metricas.csv           uma linha por modelo e teste
  resultados/fig_cnn_previsto_real.png  z previsto contra z real (teste Combo)
  resultados/fig_cnn_perda.png          perda de validacao por epoca

Uso (da raiz do repositorio):
  python src/avaliar_cnn.py
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from treinar_cnn import CNN, Conjunto  # mesma arquitetura e mesma leitura dos dados

MODELOS = Path("modelos")
RESULTADOS = Path("resultados")
TESTES = {"A": ["tz"], "B": ["gm"], "Combo": ["tz", "gm"]}
ORDEM = ["base", "tl", "lora", "combo"]
Z_MIN, Z_MAX = 0.3, 1.5
LSST_VIES, LSST_DISP = 0.003, 0.02

COR = {"tl": "#2a78d6", "lora": "#eb6834", "combo": "#1baf7a", "base": "#52514e"}
NOME = {"base": "CNN-Base", "tl": "CNN-TL", "lora": "CNN-LoRA", "combo": "CNN-Combo"}
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


def prever(modelo, conjunto, dispositivo, lote=512):
    previsto, real = [], []
    with torch.no_grad():
        for x, z in conjunto.lotes(lote, embaralhar=False):
            x = x.to(dispositivo).float()
            with torch.autocast(dispositivo.type, enabled=dispositivo.type == "cuda"):
                saida = modelo(x)
            previsto.append(saida.float().cpu().numpy())
            real.append(z.numpy())
    return np.concatenate(previsto), np.concatenate(real)


def main():
    dispositivo = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    testes = {nome: Conjunto(partes, "teste") for nome, partes in TESTES.items()}
    for nome, c in testes.items():
        print(f"teste {nome}: {c.n} galaxias")

    linhas, previsoes = [], {}
    for est in ORDEM:
        caminho = MODELOS / f"cnn_{est}.pt"
        if not caminho.exists():
            print(f"falta {caminho}, pulando")
            continue
        modelo = CNN()
        modelo.load_state_dict(torch.load(caminho, map_location="cpu")["estado"])
        modelo = modelo.to(dispositivo).eval()
        for nome, conjunto in testes.items():
            zp, z = prever(modelo, conjunto, dispositivo)
            linhas.append({"estrategia": est, "teste": nome, **metricas(zp, z)})
            if nome == "Combo":
                previsoes[est] = (zp, z)

    RESULTADOS.mkdir(exist_ok=True)
    m = pd.DataFrame(linhas)
    m.to_csv(RESULTADOS / "cnn_metricas.csv", index=False, float_format="%.5f")

    print(f"\nCNN, faixa {Z_MIN} < z < {Z_MAX}")
    print(f"{'teste':<6}{'modelo':<11}{'vies':>10}{'dispersao':>11}{'outliers %':>12}")
    for teste in TESTES:
        for _, r in m[m["teste"] == teste].iterrows():
            print(f"{teste:<6}{NOME[r['estrategia']]:<11}{r['vies']:>10.4f}"
                  f"{r['dispersao']:>11.4f}{r['outliers_pct']:>12.2f}")
    print(f"limites do LSST: |vies| < {LSST_VIES}, dispersao < {LSST_DISP}")

    treinos = RESULTADOS / "cnn_treinos.csv"
    if treinos.exists():
        t = pd.read_csv(treinos).drop_duplicates("estrategia", keep="last")
        print("\nCusto do treino")
        print(t[["estrategia", "n_treino", "parametros_treinaveis", "epocas", "minutos"]]
              .to_string(index=False))

    figura_previsto_real(previsoes)
    figura_perda()
    print(f"\nsalvo em {RESULTADOS}/: cnn_metricas.csv, fig_cnn_previsto_real.png, fig_cnn_perda.png")


def estilo(ax):
    ax.set_facecolor(FUNDO)
    for lado in ("top", "right"):
        ax.spines[lado].set_visible(False)
    for lado in ("left", "bottom"):
        ax.spines[lado].set_color(GRADE)
    ax.tick_params(colors=TINTA2, labelsize=9)
    ax.grid(axis="y", color=GRADE, linewidth=0.8)
    ax.set_axisbelow(True)


def figura_previsto_real(previsoes):
    ordem = [e for e in ORDEM if e in previsoes]
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
    fig.savefig(RESULTADOS / "fig_cnn_previsto_real.png", dpi=200, facecolor=FUNDO)
    plt.close(fig)


def figura_perda():
    """Perda de validacao por epoca. Base valida no A; TL, LoRA no B; Combo no A+B."""
    fig, ax = plt.subplots(figsize=(7, 4), facecolor=FUNDO)
    estilo(ax)
    desenhou = False
    for est in ORDEM:
        arq = RESULTADOS / f"cnn_historico_{est}.csv"
        if not arq.exists():
            continue
        h = pd.read_csv(arq)
        ax.plot(h["epoca"], h["perda_val"], color=COR[est], linewidth=2, label=NOME[est])
        desenhou = True
    if not desenhou:
        plt.close(fig)
        return
    ax.set_xlabel("Época", color=TINTA2, fontsize=9)
    ax.set_ylabel("Perda de validação", color=TINTA, fontsize=10)
    ax.legend(frameon=False, fontsize=9, labelcolor=TINTA)
    fig.text(0.5, 0.01, "Base valida no A; TL e LoRA no B; Combo no A e B juntos.",
             ha="center", color=TINTA2, fontsize=8.5)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(RESULTADOS / "fig_cnn_perda.png", dpi=200, facecolor=FUNDO)
    plt.close(fig)


if __name__ == "__main__":
    main()