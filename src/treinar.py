"""Treino da MLP com as 4 estratégias de combinação das fontes de redshift.

Estratégias:
  base   MLP-Base: treino do zero no conjunto A.
  tl     MLP-TL: parte da MLP-Base; congela tudo menos a 1ª e a 5ª camadas
         densas (TransferZ); taxa 5e-5 (decisão nossa); treino no B.
  lora   MLP-LoRA: parte da MLP-Base; pesos congelados; LoRA (peft) em todas
         as camadas lineares, posto 4, alfa 16, dropout 0,1 (artigo da LoRA);
         taxa 5e-4; treino no B.
  combo  MLP-Combo: treino do zero em A + B.

Opção A: --fracao 10, 25 ou 50 usa essa porcentagem do treino do B
(em tl, lora e combo). Sem --fracao, usa 100%.

Direção inversa: --inverso troca a ordem dos conjuntos. A base é treinada
no B (espectroscópico) e o TL e a LoRA ajustam no A (fotométrico). Não vale
para o combo (que já usa os dois) nem com --fracao. Arquivos salvos com
"_inv" no nome (ex.: modelos/base_inv_s0.pt), sem apagar os da direção normal.

Arquitetura (MLP-Base do TransferZ): 4 camadas de 200 neurônios, ligação de
atalho (a entrada é juntada à saída da 4ª camada), 2 camadas de 2000, ReLU,
1 saída. A forma exata do atalho não é descrita no artigo: decisão nossa.

Treino: Adam, lote 512, até 500 épocas; para quando a perda de validação não
melhora por 20 épocas e guarda o melhor modelo. Perda:
  L = 1 - 1 / (1 + (dz / 0,15)^2),  dz = (z_previsto - z_real) / (1 + z_real)
Normalização: média e desvio do treino de cada etapa (TransferZ).

Entradas: data/modelo/*.csv (gerados pelo preparar_dados.py)
Saídas:
  modelos/<estrategia>[_f<fracao>]_s<semente>.pt
  resultados/treinos.csv  (uma linha por treino)

Uso (da raiz do repositório):
  python src/treinar.py base
  python src/treinar.py tl
  python src/treinar.py lora --fracao 25
  python src/treinar.py combo --semente 1
  python src/treinar.py base --inverso --semente 0
"""
import argparse
import csv
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

BANDAS = ["g", "r", "i", "z", "y"]
DADOS = Path("data/modelo")
MODELOS = Path("modelos")
RESULTADOS = Path("resultados")
LOTE = 512
EPOCAS_MAX = 500
PACIENCIA = 20
TAXA = {"base": 5e-4, "tl": 5e-5, "lora": 5e-4, "combo": 5e-4}
CAMADAS_LINEARES = ["c1", "c2", "c3", "c4", "c5", "c6", "saida"]


class MLP(nn.Module):
    def __init__(self, n_entrada=5):
        super().__init__()
        self.c1 = nn.Linear(n_entrada, 200)
        self.c2 = nn.Linear(200, 200)
        self.c3 = nn.Linear(200, 200)
        self.c4 = nn.Linear(200, 200)
        self.c5 = nn.Linear(200 + n_entrada, 2000)   # recebe a 4ª camada + a entrada (atalho)
        self.c6 = nn.Linear(2000, 2000)
        self.saida = nn.Linear(2000, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        h = self.relu(self.c1(x))
        h = self.relu(self.c2(h))
        h = self.relu(self.c3(h))
        h = self.relu(self.c4(h))
        h = torch.cat([h, x], dim=1)
        h = self.relu(self.c5(h))
        h = self.relu(self.c6(h))
        return self.saida(h).squeeze(1)


def perda(z_prev, z_real):
    dz = (z_prev - z_real) / (1 + z_real)
    return (1 - 1 / (1 + (dz / 0.15) ** 2)).mean()


def carregar(nome):
    df = pd.read_csv(DADOS / f"{nome}.csv")
    return df[BANDAS].to_numpy(np.float32), df["redshift"].to_numpy(np.float32)


def arquivos(estrategia, fracao, inverso=False):
    """Quais arquivos de treino e validação cada estratégia usa."""
    if inverso:   # base no B; ajuste no A
        return (["b_treino"], "b_val") if estrategia == "base" else (["a_treino"], "a_val")
    sufixo = "" if fracao == 100 else f"_{fracao}"
    if estrategia == "base":
        return ["a_treino"], "a_val"
    if estrategia in ("tl", "lora"):
        return [f"b_treino{sufixo}"], "b_val"
    return ["a_treino", f"b_treino{sufixo}"], "combo_val"   # combo


def avaliar_perda(modelo, x, y):
    modelo.eval()
    with torch.no_grad():
        total = 0.0
        for i in range(0, len(x), 8192):
            total += perda(modelo(x[i:i + 8192]), y[i:i + 8192]).item() * len(x[i:i + 8192])
    return total / len(x)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("estrategia", choices=["base", "tl", "lora", "combo"])
    p.add_argument("--fracao", type=int, default=100, choices=[10, 25, 50, 100])
    p.add_argument("--semente", type=int, default=0)
    p.add_argument("--inverso", action="store_true", help="base no B, ajuste no A")
    a = p.parse_args()
    if a.estrategia == "base" and a.fracao != 100:
        p.error("a fração só vale para tl, lora e combo")
    if a.inverso and (a.estrategia == "combo" or a.fracao != 100):
        p.error("--inverso só vale para base, tl e lora, com 100% dos dados")
    inv = "_inv" if a.inverso else ""

    torch.manual_seed(a.semente)
    np.random.seed(a.semente)
    dispositivo = "cuda" if torch.cuda.is_available() else "cpu"

    # dados
    nomes_treino, nome_val = arquivos(a.estrategia, a.fracao, a.inverso)
    partes = [carregar(n) for n in nomes_treino]
    x_tr = np.concatenate([q[0] for q in partes])
    y_tr = np.concatenate([q[1] for q in partes])
    x_va, y_va = carregar(nome_val)
    media, desvio = x_tr.mean(axis=0), x_tr.std(axis=0)   # normalização desta etapa
    x_tr = torch.tensor((x_tr - media) / desvio, device=dispositivo)
    x_va = torch.tensor((x_va - media) / desvio, device=dispositivo)
    y_tr = torch.tensor(y_tr, device=dispositivo)
    y_va = torch.tensor(y_va, device=dispositivo)
    print(f"{a.estrategia}{inv}, fração {a.fracao}%, semente {a.semente}, {dispositivo}")
    print(f"treino: {'+'.join(nomes_treino)} ({len(x_tr)}) | validação: {nome_val} ({len(x_va)})")

    # modelo
    modelo = MLP()
    if a.estrategia in ("tl", "lora"):
        base = torch.load(MODELOS / f"base{inv}_s{a.semente}.pt", map_location="cpu", weights_only=False)
        modelo.load_state_dict(base["estado"])
    if a.estrategia == "tl":
        for nome, par in modelo.named_parameters():
            par.requires_grad = nome.startswith(("c1.", "c5."))
    if a.estrategia == "lora":
        from peft import LoraConfig, get_peft_model
        cfg = LoraConfig(r=4, lora_alpha=16, lora_dropout=0.1, target_modules=CAMADAS_LINEARES)
        modelo = get_peft_model(modelo, cfg)   # congela os pesos da base e cria os adaptadores
    modelo.to(dispositivo)
    treinaveis = sum(q.numel() for q in modelo.parameters() if q.requires_grad)
    total_par = sum(q.numel() for q in modelo.parameters())
    print(f"parâmetros treináveis: {treinaveis} de {total_par}")

    otim = torch.optim.Adam([q for q in modelo.parameters() if q.requires_grad], lr=TAXA[a.estrategia])
    gerador = torch.Generator(device="cpu").manual_seed(a.semente)

    melhor, melhor_estado, sem_melhora, epoca = float("inf"), None, 0, 0
    inicio = time.time()
    for epoca in range(1, EPOCAS_MAX + 1):
        modelo.train()
        ordem = torch.randperm(len(x_tr), generator=gerador).to(dispositivo)
        for i in range(0, len(x_tr), LOTE):
            idx = ordem[i:i + LOTE]
            otim.zero_grad()
            l = perda(modelo(x_tr[idx]), y_tr[idx])
            l.backward()
            otim.step()
        val = avaliar_perda(modelo, x_va, y_va)
        if val < melhor:
            melhor, sem_melhora = val, 0
            melhor_estado = {k: v.detach().clone() for k, v in modelo.state_dict().items()}
        else:
            sem_melhora += 1
        if epoca == 1 or epoca % 10 == 0:
            print(f"  época {epoca:3d}: validação {val:.5f} (melhor {melhor:.5f}), "
                  f"{(time.time() - inicio) / 60:.1f} min")
        if sem_melhora >= PACIENCIA:
            print(f"  parou na época {epoca}: {PACIENCIA} épocas sem melhora")
            break
    minutos = (time.time() - inicio) / 60

    # guarda o melhor modelo como MLP comum (no LoRA, junta os adaptadores aos pesos)
    modelo.load_state_dict(melhor_estado)
    if a.estrategia == "lora":
        modelo = modelo.merge_and_unload()
    sufixo = "" if a.fracao == 100 else f"_f{a.fracao}"
    MODELOS.mkdir(exist_ok=True)
    arq = MODELOS / f"{a.estrategia}{sufixo}{inv}_s{a.semente}.pt"
    torch.save({"estado": {k: v.cpu() for k, v in modelo.state_dict().items()},
                "media": media, "desvio": desvio, "estrategia": a.estrategia,
                "fracao": a.fracao, "semente": a.semente, "inverso": a.inverso}, arq)

    RESULTADOS.mkdir(exist_ok=True)
    log = RESULTADOS / "treinos.csv"
    novo = not log.exists()
    with open(log, "a", newline="") as f:
        w = csv.writer(f)
        if novo:
            w.writerow(["estrategia", "fracao", "semente", "n_treino", "epocas", "melhor_validacao",
                        "minutos", "parametros_treinaveis", "arquivo"])
        w.writerow([a.estrategia + inv, a.fracao, a.semente, len(x_tr), epoca, f"{melhor:.6f}",
                    f"{minutos:.2f}", treinaveis, arq.as_posix()])
    print(f"melhor validação {melhor:.5f}, {epoca} épocas, {minutos:.1f} min, salvo em {arq}")


if __name__ == "__main__":
    main()