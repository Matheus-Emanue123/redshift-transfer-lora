"""Raspagem do catálogo HSC-SSP PDR2 (COSMOS wide-depth, seeing mediano).

Usa a mesma API da ferramenta oficial hscReleaseQuery.py (versão 20190514.1):
submit -> status -> download, em
https://hsc-release.mtk.nao.ac.jp/datasearch/api/catalog_jobs/
"""
import getpass
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

API_URL = "https://hsc-release.mtk.nao.ac.jp/datasearch/api/catalog_jobs/"
CLIENT_VERSION = 20190514.1          # mesma versão da ferramenta oficial
RELEASE = "pdr2"
FORMATO = "csv.gz"
SAIDA = Path("data/raw/hsc_pdr2_cosmos_median.csv.gz")

SQL = """
SELECT
  object_id, ra, dec, isprimary,
  g_cmodel_mag, r_cmodel_mag, i_cmodel_mag, z_cmodel_mag, y_cmodel_mag,
  g_cmodel_magsigma, r_cmodel_magsigma, i_cmodel_magsigma, z_cmodel_magsigma, y_cmodel_magsigma,
  i_extendedness_value,
  a_g, a_r, a_i, a_z, a_y
FROM pdr2_cosmos_wide_depth_median.forced
"""


def post(endpoint, dados):
    dados["clientVersion"] = CLIENT_VERSION
    req = urllib.request.Request(
        API_URL + endpoint,
        json.dumps(dados).encode("utf-8"),
        {"Content-type": "application/json"},
    )
    return urllib.request.urlopen(req)


def credencial():
    usuario = os.environ.get("HSC_SSP_CAS_USER") or input("usuário HSC: ")
    senha = os.environ.get("HSC_SSP_CAS_PASSWORD") or getpass.getpass("senha HSC: ")
    return {"account_name": usuario, "password": senha}


def preview(cred):
    """Roda a consulta com LIMIT 10 no modo rápido, só para testar os nomes."""
    sql = SQL if "limit" in SQL.lower() else SQL.rstrip().rstrip(";") + " LIMIT 10"
    job = {"sql": sql, "release_version": RELEASE}
    res = json.load(post("preview", {"credential": cred, "catalog_job": job}))
    for linha in res["result"]["rows"]:
        print(linha)


def coletar(cred):
    job = {
        "sql": SQL,
        "out_format": FORMATO,
        "include_metainfo_to_body": True,   # igual à ferramenta oficial
        "release_version": RELEASE,
    }
    dados = {"credential": cred, "catalog_job": job,
             "nomail": True, "skip_syntax_check": False}
    job_id = json.load(post("submit", dados))["id"]
    print("job enviado, id:", job_id)

    # espera o servidor terminar (intervalo dobra até 5 min, como na oficial)
    intervalo = 1
    while True:
        time.sleep(intervalo)
        status = json.load(post("status", {"credential": cred, "id": job_id}))
        print("status:", status["status"])
        if status["status"] == "error":
            raise RuntimeError("erro na consulta: " + status["error"])
        if status["status"] == "done":
            break
        intervalo = min(intervalo * 2, 300)

    # baixa direto para o arquivo, em blocos de 64 KB
    SAIDA.parent.mkdir(parents=True, exist_ok=True)
    res = post("download", {"credential": cred, "id": job_id})
    total = 0
    with open(SAIDA, "wb") as f:
        while True:
            bloco = res.read(64 * 1024)
            if not bloco:
                break
            f.write(bloco)
            total += len(bloco)
    print(f"{total / 1e6:.1f} MB salvos em {SAIDA}")


if __name__ == "__main__":
    # --sql arquivo.sql troca a consulta padrão (útil para verificações)
    if "--sql" in sys.argv:
        SQL = Path(sys.argv[sys.argv.index("--sql") + 1]).read_text(encoding="utf-8")
    # --saida arquivo.csv.gz troca o arquivo de saída da coleta completa
    if "--saida" in sys.argv:
        SAIDA = Path(sys.argv[sys.argv.index("--saida") + 1])
    cred = credencial()
    try:
        if "--preview" in sys.argv:
            preview(cred)
        else:
            coletar(cred)
    except urllib.error.HTTPError as e:
        # mostra o código e a resposta completa do servidor
        print("HTTP", e.code, e.reason)
        print(e.read().decode(errors="ignore"))
        sys.exit(1)