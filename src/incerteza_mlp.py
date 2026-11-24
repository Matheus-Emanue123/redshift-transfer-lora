"""Incerteza da MLP pelo conjunto das 3 sementes (sem treino novo).

Para cada estrategia (base, tl, lora, combo) com 100% do B, carrega os 3 modelos
(sementes 0, 1 e 2) e preve o redshift de cada galaxia dos 3 testes.
  z_medio = media das 3 previsoes
  sigma   = desvio padrao das 3 previsoes   (incerteza estimada)

Perguntas:
  1. A incerteza acompanha o erro? Correlacao de Spearman entre sigma e |dz|,
     e dispersao de dz em cada quinto de sigma (do menor para o maior).
  2. Os intervalos acertam a porcentagem esperada? Fracao de galaxias com
     |z_medio - z_real| <= 1 sigma (esperado 68,3% se o erro fosse normal)
     e <= 2 sigma (esperado 95,4%). Sigma em unidades de z.
  3. A media das 3 previsoes e melhor que um modelo sozinho?

Faixa 0,3 < z_real < 1,5, como nas outras avaliacoes.
Atencao: o desvio de so 3 previsoes e uma estimativa grosseira, e mede so a
incerteza do modelo, nao o erro dos rotulos.

Saidas:
  resultados/incerteza_mlp.csv
  resultados/fig_incerteza_mlp.png

Uso (da raiz do repositorio):
  python src/incerteza_mlp.py
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr

from avaliar import prever, metricas, estilo, DADOS, TESTES, Z_MIN, Z_MAX, COR, NOME, TINTA, TINTA2, FUNDO
from treinar import BANDAS

MODELOS = Path("modelos")
RESULTADOS = Path("resultados")
ESTRATEGIAS = ["base", "tl", "lora", "combo"]
SEMENTES = [0, 1, 2]


def main():
    testes = {}
    for nome, arq in TESTES.items():
        df = pd.read_csv(DADOS / f"{arq}.csv")
        testes[nome] = (df[BANDAS].to_numpy(np.float32), df["redshift"].to_numpy(np.float32))

    linhas, quintis = [], {}
    for est in ESTRATEGIAS:
        infos = [torch.load(MODELOS / f"{est}_s{s}.pt", map_location="cpu", weights_only=False)
                 for s in SEMENTES]
        for nome, (x, z) in testes.items():
            prev = np.stack([prever(info, x) for info in infos])  # 3 x n
            z_medio, sigma = prev.mean(0), prev.std(0, ddof=1)

            faixa = (z > Z_MIN) & (z < Z_MAX)
            zm, sg, zr = z_medio[faixa], sigma[faixa], z[faixa]
            dz = (zm - zr) / (1 + zr)
            erro = np.abs(zm - zr)
            rho = spearmanr(sg, np.abs(dz)).correlation

            # dispersao de dz em cada quinto de sigma
            cortes = np.quantile(sg, [0.2, 0.4, 0.6, 0.8])
            grupo = np.digitize(sg, cortes)
            disp_q = []
            for g in range(5):
                d = dz[grupo == g]
                disp_q.append(1.4826 * np.median(np.abs(d - np.median(d))))
            quintis[(est, nome)] = disp_q

            individual = np.mean([metricas(p, z)["dispersao"] for p in prev])
            linhas.append({
                "estrategia": est, "teste": nome,
                "dispersao_media_3": metricas(z_medio, z)["dispersao"],
                "dispersao_individual": individual,
                "sigma_mediano": np.median(sg),
                "spearman_sigma_erro": rho,
                "cobertura_1sigma_pct": 100 * np.mean(erro <= sg),
                "cobertura_2sigma_pct": 100 * np.mean(erro <= 2 * sg),
                "disp_quinto_menor_sigma": disp_q[0],
                "disp_quinto_maior_sigma": disp_q[4],
            })

    RESULTADOS.mkdir(exist_ok=True)
    m = pd.DataFrame(linhas)
    m.to_csv(RESULTADOS / "incerteza_mlp.csv", index=False, float_format="%.5f")

    print(f"Incerteza pelas 3 sementes, faixa {Z_MIN} < z < {Z_MAX}")
    print(f"{'teste':<6}{'modelo':<11}{'sigma med':>10}{'Spearman':>10}{'cob 1s %':>10}"
          f"{'cob 2s %':>10}{'disp Q1':>9}{'disp Q5':>9}{'disp media3':>12}{'disp indiv':>11}")
    for teste in TESTES:
        for _, r in m[m["teste"] == teste].iterrows():
            print(f"{teste:<6}{NOME[r['estrategia']]:<11}{r['sigma_mediano']:>10.4f}"
                  f"{r['spearman_sigma_erro']:>10.3f}{r['cobertura_1sigma_pct']:>10.1f}"
                  f"{r['cobertura_2sigma_pct']:>10.1f}{r['disp_quinto_menor_sigma']:>9.4f}"
                  f"{r['disp_quinto_maior_sigma']:>9.4f}{r['dispersao_media_3']:>12.4f}"
                  f"{r['dispersao_individual']:>11.4f}")
    print("esperado se o erro fosse normal: cobertura 68,3% (1 sigma) e 95,4% (2 sigma)")

    figura(quintis)
    print(f"\nsalvo em {RESULTADOS}/: incerteza_mlp.csv, fig_incerteza_mlp.png")


def figura(quintis):
    fig, eixos = plt.subplots(1, len(TESTES), figsize=(11, 3.6), facecolor=FUNDO)
    pos = np.arange(5)
    for ax, teste in zip(eixos, TESTES):
        estilo(ax)
        for est in ESTRATEGIAS:
            ax.plot(pos, quintis[(est, teste)], color=COR[est], linewidth=2, marker="o",
                    markersize=7, markeredgecolor=FUNDO, markeredgewidth=1.5, label=NOME[est])
        ax.set_xticks(pos, ["1º", "2º", "3º", "4º", "5º"])
        ax.set_title(f"Teste {teste}", color=TINTA, fontsize=11)
        ax.set_xlabel("Quinto de sigma (menor → maior)", color=TINTA2, fontsize=9)
    eixos[0].set_ylabel("Dispersão de dz", color=TINTA, fontsize=10)
    alcas, rotulos = eixos[0].get_legend_handles_labels()
    fig.legend(alcas, rotulos, loc="upper center", ncol=4, frameon=False, fontsize=10, labelcolor=TINTA)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(RESULTADOS / "fig_incerteza_mlp.png", dpi=200, facecolor=FUNDO)
    plt.close(fig)


if __name__ == "__main__":
    main()