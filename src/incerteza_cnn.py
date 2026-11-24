"""Incerteza da CNN por MC dropout (sem treino novo).

O regressor da CNN tem duas camadas de dropout (0,5). Na avaliacao normal o
dropout fica desligado. Aqui ele fica ligado e a previsao e repetida 30 vezes,
cada vez com neuronios diferentes desligados:
  z_medio = media das 30 previsoes
  sigma   = desvio padrao das 30 previsoes   (incerteza estimada)
A parte ResNet nao tem dropout, entao ela roda uma vez so por galaxia e so o
regressor e repetido. As camadas BatchNorm ficam no modo de avaliacao.

Mesmas perguntas do incerteza_mlp.py:
  1. Correlacao de Spearman entre sigma e |dz|; dispersao no quinto de menor e
     de maior sigma.
  2. Cobertura: % de galaxias com |z_medio - z_real| <= 1 sigma (esperado
     68,3%) e <= 2 sigma (esperado 95,4%).
  3. Dispersao de z_medio comparada com a previsao normal (dropout desligado).

Faixa 0,3 < z_real < 1,5. Testes A (TransferZ-Images sem as repetidas),
B (GalaxiesML) e Combo (A e B juntos).

Saidas:
  resultados/incerteza_cnn.csv
  resultados/fig_incerteza_cnn.png

Uso (da raiz do repositorio):
  python src/incerteza_cnn.py
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr

from treinar_cnn import CNN, Conjunto
from avaliar_cnn import metricas, estilo, Z_MIN, Z_MAX, COR, NOME, ORDEM, TINTA, TINTA2, FUNDO

MODELOS = Path("modelos")
RESULTADOS = Path("resultados")
N_REPETICOES = 30
SEMENTE = 42


def prever_mc(modelo, conjunto, dispositivo, lote=512):
    """Devolve (previsao normal, previsoes com dropout [N x n], z real)."""
    modelo.eval()
    normal, mc, real = [], [], []
    with torch.no_grad():
        for x, z in conjunto.lotes(lote, embaralhar=False):
            x = x.to(dispositivo).float()
            with torch.autocast(dispositivo.type, enabled=dispositivo.type == "cuda"):
                f = modelo.features(x).flatten(1)
            f = f.float()
            modelo.classifier.eval()
            normal.append(modelo.classifier(f).squeeze(1).cpu().numpy())
            modelo.classifier.train()  # liga so o dropout do regressor
            mc.append(torch.stack([modelo.classifier(f).squeeze(1)
                                   for _ in range(N_REPETICOES)]).cpu().numpy())
            modelo.classifier.eval()
            real.append(z.numpy())
    return np.concatenate(normal), np.concatenate(mc, axis=1), np.concatenate(real)


def resumo(normal, mc, z):
    z_medio, sigma = mc.mean(0), mc.std(0, ddof=1)
    faixa = (z > Z_MIN) & (z < Z_MAX)
    zm, sg, zr = z_medio[faixa], sigma[faixa], z[faixa]
    dz = (zm - zr) / (1 + zr)
    erro = np.abs(zm - zr)
    cortes = np.quantile(sg, [0.2, 0.4, 0.6, 0.8])
    grupo = np.digitize(sg, cortes)
    disp_q = []
    for g in range(5):
        d = dz[grupo == g]
        disp_q.append(1.4826 * np.median(np.abs(d - np.median(d))))
    return {
        "dispersao_media_mc": metricas(z_medio, z)["dispersao"],
        "dispersao_normal": metricas(normal, z)["dispersao"],
        "sigma_mediano": np.median(sg),
        "spearman_sigma_erro": spearmanr(sg, np.abs(dz)).correlation,
        "cobertura_1sigma_pct": 100 * np.mean(erro <= sg),
        "cobertura_2sigma_pct": 100 * np.mean(erro <= 2 * sg),
        "disp_quinto_menor_sigma": disp_q[0],
        "disp_quinto_maior_sigma": disp_q[4],
    }, disp_q


def main():
    torch.manual_seed(SEMENTE)
    dispositivo = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    conjuntos = {"A": Conjunto(["tz"], "teste"), "B": Conjunto(["gm"], "teste")}

    linhas, quintis = [], {}
    for est in ORDEM:
        modelo = CNN()
        modelo.load_state_dict(torch.load(MODELOS / f"cnn_{est}.pt", map_location="cpu")["estado"])
        modelo = modelo.to(dispositivo)
        saidas = {nome: prever_mc(modelo, c, dispositivo) for nome, c in conjuntos.items()}
        # teste Combo = A e B juntos (mesmas galaxias do avaliar_cnn.py)
        saidas["Combo"] = tuple(np.concatenate([saidas["A"][k], saidas["B"][k]], axis=-1)
                                for k in range(3))
        for nome, (normal, mc, z) in saidas.items():
            r, disp_q = resumo(normal, mc, z)
            linhas.append({"estrategia": est, "teste": nome, **r})
            quintis[(est, nome)] = disp_q
        print(f"{NOME[est]} pronto", flush=True)

    RESULTADOS.mkdir(exist_ok=True)
    m = pd.DataFrame(linhas)
    m.to_csv(RESULTADOS / "incerteza_cnn.csv", index=False, float_format="%.5f")

    print(f"\nIncerteza por MC dropout ({N_REPETICOES} repeticoes), faixa {Z_MIN} < z < {Z_MAX}")
    print(f"{'teste':<6}{'modelo':<11}{'sigma med':>10}{'Spearman':>10}{'cob 1s %':>10}"
          f"{'cob 2s %':>10}{'disp Q1':>9}{'disp Q5':>9}{'disp MC':>9}{'disp normal':>12}")
    for teste in ["A", "B", "Combo"]:
        for _, r in m[m["teste"] == teste].iterrows():
            print(f"{teste:<6}{NOME[r['estrategia']]:<11}{r['sigma_mediano']:>10.4f}"
                  f"{r['spearman_sigma_erro']:>10.3f}{r['cobertura_1sigma_pct']:>10.1f}"
                  f"{r['cobertura_2sigma_pct']:>10.1f}{r['disp_quinto_menor_sigma']:>9.4f}"
                  f"{r['disp_quinto_maior_sigma']:>9.4f}{r['dispersao_media_mc']:>9.4f}"
                  f"{r['dispersao_normal']:>12.4f}")
    print("esperado se o erro fosse normal: cobertura 68,3% (1 sigma) e 95,4% (2 sigma)")

    figura(quintis)
    print(f"\nsalvo em {RESULTADOS}/: incerteza_cnn.csv, fig_incerteza_cnn.png")


def figura(quintis):
    fig, eixos = plt.subplots(1, 3, figsize=(11, 3.6), facecolor=FUNDO)
    pos = np.arange(5)
    for ax, teste in zip(eixos, ["A", "B", "Combo"]):
        estilo(ax)
        for est in ORDEM:
            ax.plot(pos, quintis[(est, teste)], color=COR[est], linewidth=2, marker="o",
                    markersize=7, markeredgecolor=FUNDO, markeredgewidth=1.5, label=NOME[est])
        ax.set_xticks(pos, ["1º", "2º", "3º", "4º", "5º"])
        ax.set_title(f"Teste {teste}", color=TINTA, fontsize=11)
        ax.set_xlabel("Quinto de sigma (menor → maior)", color=TINTA2, fontsize=9)
    eixos[0].set_ylabel("Dispersão de dz", color=TINTA, fontsize=10)
    alcas, rotulos = eixos[0].get_legend_handles_labels()
    fig.legend(alcas, rotulos, loc="upper center", ncol=4, frameon=False, fontsize=10, labelcolor=TINTA)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(RESULTADOS / "fig_incerteza_cnn.png", dpi=200, facecolor=FUNDO)
    plt.close(fig)


if __name__ == "__main__":
    main()