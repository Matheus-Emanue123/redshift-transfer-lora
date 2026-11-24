"""Treina a CNN (ResNet18 com 5 canais) para estimar redshift a partir das imagens.

Uso:
    python src/treinar_cnn.py base     # treina com TransferZ-Images (tz)
    python src/treinar_cnn.py tl       # parte do base, treina com GalaxiesML (gm)
    python src/treinar_cnn.py lora     # parte do base, treina adaptador LoRA com gm
    python src/treinar_cnn.py combo    # do zero, com tz + gm, mesmo numero de epocas do lora

Ordem obrigatoria: base, depois tl e lora, e por ultimo combo.

Configuracao do artigo LoRA (Seenivasan et al., codigo astrodatalab/seenivasan_2025):
ResNet18 sem pesos pre-treinados, primeira camada com 5 canais, regressor
512 -> 256 -> 1 com SiLU e dropout 0,5, Adam com weight_decay 1e-4,
ReduceLROnPlateau (fator 0,1, paciencia 5), corte de gradiente 1,0,
redshift limitado a [0, 4], parada pela media movel exponencial (EMA) da perda
de validacao. LoRA: r=4, alpha=16, dropout 0,1, em todas as Conv2d e Linear.

Mudancas por causa do prazo e da GPU de 4 GB (ver mensagem do Passo 2):
maximo de 50 epocas e paciencia 10 (artigo: 500 e 20); TL com lr 1e-4
(artigo: 1e-8 por 500 epocas); as 557 galaxias repetidas sao retiradas do tz.
"""
import argparse
import csv
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models

DADOS = Path("data/imagens")
MODELOS = Path("modelos")
RESULTADOS = Path("resultados")
SEMENTE = 42
ALFA_EMA = 2 / 11  # mesmo valor do codigo dos autores

CONFIG = {
    "base": {"lr": 1e-3, "treino": ["tz"]},
    "tl": {"lr": 1e-4, "treino": ["gm"]},
    "lora": {"lr": 1e-4, "treino": ["gm"]},
    "combo": {"lr": 1e-3, "treino": ["tz", "gm"]},
}

ALVOS_LORA = [
    "features.0",
    "features.4.0.conv1", "features.4.0.conv2", "features.4.1.conv1", "features.4.1.conv2",
    "features.5.0.conv1", "features.5.0.conv2", "features.5.1.conv1", "features.5.1.conv2",
    "features.6.0.conv1", "features.6.0.conv2", "features.6.1.conv1", "features.6.1.conv2",
    "features.7.0.conv1", "features.7.0.conv2", "features.7.1.conv1", "features.7.1.conv2",
    "features.5.0.downsample.0", "features.6.0.downsample.0", "features.7.0.downsample.0",
    "classifier.0", "classifier.3", "classifier.6",
]


class CNN(nn.Module):
    """ResNet18 com entrada de 5 canais e regressor 512 -> 256 -> 1."""

    def __init__(self):
        super().__init__()
        resnet = models.resnet18(weights=None)
        resnet.conv1 = nn.Conv2d(5, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.features = nn.Sequential(*list(resnet.children())[:-1])
        self.classifier = nn.Sequential(
            nn.Linear(512, 512), nn.SiLU(), nn.Dropout(0.5),
            nn.Linear(512, 256), nn.SiLU(), nn.Dropout(0.5),
            nn.Linear(256, 1),
        )

    def forward(self, x):
        return self.classifier(self.features(x).flatten(1)).squeeze(1)


def perda(z_prev, z_real):
    dz = (z_prev - z_real) / (1 + z_real)
    return torch.mean(1 - 1 / (1 + (dz / 0.15) ** 2))


class Conjunto:
    """Junta um ou mais conjuntos (tz, gm). As imagens ficam no disco (memmap)."""

    def __init__(self, nomes, parte):
        repetidas = np.load(DADOS / "repetidas_id.npy")
        self.partes = []
        for nome in nomes:
            img = np.load(DADOS / f"{nome}_{parte}_img.npy", mmap_mode="r")
            z = np.load(DADOS / f"{nome}_{parte}_z.npy")
            ids = np.load(DADOS / f"{nome}_{parte}_id.npy")
            manter = np.arange(len(z))
            if nome == "tz":  # nas repetidas, fica a versao do GalaxiesML
                manter = np.flatnonzero(~np.isin(ids, repetidas))
            self.partes.append((img, z, manter))
        self.n = sum(len(m) for _, _, m in self.partes)

    def lotes(self, tamanho, embaralhar, rng=None):
        ordem = [(k, i) for k, (_, _, m) in enumerate(self.partes) for i in m]
        ordem = np.array(ordem)
        if embaralhar:
            ordem = ordem[rng.permutation(len(ordem))]
        for inicio in range(0, len(ordem), tamanho):
            pedaco = ordem[inicio:inicio + tamanho]
            imgs, zs = [], []
            for k, (img, z, _) in enumerate(self.partes):
                idx = np.sort(pedaco[pedaco[:, 0] == k, 1])
                if len(idx):
                    imgs.append(img[idx])
                    zs.append(z[idx])
            yield (torch.from_numpy(np.concatenate(imgs)),
                   torch.from_numpy(np.concatenate(zs)))


def avaliar_perda(modelo, conjunto, dispositivo, lote):
    modelo.eval()
    soma, n = 0.0, 0
    with torch.no_grad():
        for x, z in conjunto.lotes(lote, embaralhar=False):
            x = x.to(dispositivo).float()
            z = z.to(dispositivo).float()
            with torch.autocast(dispositivo.type, enabled=dispositivo.type == "cuda"):
                saida = modelo(x)
            soma += perda(saida.float(), z).item() * len(z)
            n += len(z)
    return soma / n


def carregar_base():
    caminho = MODELOS / "cnn_base.pt"
    if not caminho.exists():
        raise SystemExit("Rode primeiro: python src/treinar_cnn.py base")
    modelo = CNN()
    modelo.load_state_dict(torch.load(caminho, map_location="cpu")["estado"])
    return modelo


def epocas_do_lora():
    caminho = MODELOS / "cnn_lora.pt"
    if not caminho.exists():
        raise SystemExit("Rode primeiro: python src/treinar_cnn.py lora")
    return torch.load(caminho, map_location="cpu")["epocas"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("estrategia", choices=list(CONFIG))
    ap.add_argument("--max_epocas", type=int, default=50)
    ap.add_argument("--paciencia", type=int, default=10)
    ap.add_argument("--lote", type=int, default=512)
    args = ap.parse_args()

    torch.manual_seed(SEMENTE)
    rng = np.random.default_rng(SEMENTE)
    dispositivo = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = CONFIG[args.estrategia]

    # Modelo e parametros treinaveis
    if args.estrategia == "base" or args.estrategia == "combo":
        modelo = CNN()
    elif args.estrategia == "tl":
        modelo = carregar_base()
        for nome, p in modelo.named_parameters():
            p.requires_grad = nome.startswith("features.0.") or nome.startswith("classifier.")
    else:  # lora
        from peft import LoraConfig, get_peft_model
        modelo = get_peft_model(carregar_base(), LoraConfig(
            r=4, lora_alpha=16, lora_dropout=0.1, target_modules=ALVOS_LORA, bias="none"))
    modelo = modelo.to(dispositivo)
    treinaveis = [p for p in modelo.parameters() if p.requires_grad]
    n_treinaveis = sum(p.numel() for p in treinaveis)

    # Combo: mesmo numero de epocas do LoRA, sem parada antecipada (como no artigo)
    max_epocas, parar_cedo = args.max_epocas, True
    if args.estrategia == "combo":
        max_epocas, parar_cedo = epocas_do_lora(), False

    treino = Conjunto(cfg["treino"], "treino")
    val = Conjunto(cfg["treino"], "val")
    print(f"{args.estrategia}: {treino.n} galaxias de treino, {val.n} de validacao, "
          f"{n_treinaveis} parametros treinaveis, dispositivo {dispositivo}")

    otimizador = torch.optim.Adam(treinaveis, lr=cfg["lr"], weight_decay=1e-4)
    agendador = torch.optim.lr_scheduler.ReduceLROnPlateau(otimizador, factor=0.1, patience=5)
    escala = torch.amp.GradScaler(enabled=dispositivo.type == "cuda")

    MODELOS.mkdir(exist_ok=True)
    RESULTADOS.mkdir(exist_ok=True)
    historico = []
    melhor_val, melhor_estado, melhor_epoca = float("inf"), None, 0
    melhor_ema, sem_melhora, ema = float("inf"), 0, None
    inicio_total = time.time()

    for epoca in range(1, max_epocas + 1):
        t0 = time.time()
        modelo.train()
        soma, n = 0.0, 0
        for x, z in treino.lotes(args.lote, embaralhar=True, rng=rng):
            x = x.to(dispositivo, non_blocking=True).float()
            z = z.to(dispositivo).float().clamp(0, 4)
            otimizador.zero_grad()
            with torch.autocast(dispositivo.type, enabled=dispositivo.type == "cuda"):
                saida = modelo(x)
            l = perda(saida.float(), z)
            escala.scale(l).backward()
            escala.unscale_(otimizador)
            torch.nn.utils.clip_grad_norm_(treinaveis, max_norm=1.0)
            escala.step(otimizador)
            escala.update()
            soma += l.item() * len(z)
            n += len(z)
        perda_treino = soma / n
        perda_val = avaliar_perda(modelo, val, dispositivo, args.lote)
        agendador.step(perda_val)
        ema = perda_val if ema is None else ALFA_EMA * ema + (1 - ALFA_EMA) * perda_val
        lr = otimizador.param_groups[0]["lr"]
        segundos = time.time() - t0
        historico.append([epoca, perda_treino, perda_val, ema, lr, segundos])
        print(f"  epoca {epoca:3d}  treino {perda_treino:.4f}  val {perda_val:.4f}  "
              f"ema {ema:.4f}  lr {lr:.0e}  {segundos:.0f} s", flush=True)

        if perda_val < melhor_val:
            melhor_val, melhor_epoca = perda_val, epoca
            melhor_estado = {k: v.detach().cpu().clone() for k, v in modelo.state_dict().items()}
        if ema < melhor_ema:
            melhor_ema, sem_melhora = ema, 0
        else:
            sem_melhora += 1
        if parar_cedo and sem_melhora >= args.paciencia:
            print(f"  parada antecipada na epoca {epoca}")
            break

    epocas = len(historico)
    if args.estrategia == "combo":  # sem parada antecipada: guarda a ultima epoca
        melhor_estado = {k: v.detach().cpu().clone() for k, v in modelo.state_dict().items()}
        melhor_epoca, melhor_val = epocas, historico[-1][2]
    modelo.load_state_dict(melhor_estado)
    if args.estrategia == "lora":  # junta o adaptador aos pesos (mesmo formato do base)
        modelo = modelo.merge_and_unload()
    estado = {k: v.detach().cpu() for k, v in modelo.state_dict().items()}
    minutos = (time.time() - inicio_total) / 60

    torch.save({"estado": estado, "epocas": epocas, "melhor_epoca": melhor_epoca},
               MODELOS / f"cnn_{args.estrategia}.pt")
    with open(RESULTADOS / f"cnn_historico_{args.estrategia}.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["epoca", "perda_treino", "perda_val", "ema_val", "lr", "segundos"])
        w.writerows(historico)
    resumo = RESULTADOS / "cnn_treinos.csv"
    novo = not resumo.exists()
    with open(resumo, "a", newline="") as f:
        w = csv.writer(f)
        if novo:
            w.writerow(["estrategia", "n_treino", "parametros_treinaveis", "epocas",
                        "melhor_epoca", "melhor_perda_val", "minutos"])
        w.writerow([args.estrategia, treino.n, n_treinaveis, epocas, melhor_epoca,
                    f"{melhor_val:.5f}", f"{minutos:.1f}"])
    print(f"Fim: {epocas} epocas, melhor na epoca {melhor_epoca} "
          f"(perda val {melhor_val:.4f}), {minutos:.1f} min")


if __name__ == "__main__":
    main()