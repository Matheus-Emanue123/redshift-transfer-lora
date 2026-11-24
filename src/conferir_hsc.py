import gzip
import pandas as pd

ARQ = "data/raw/hsc_pdr2_cosmos_median.csv.gz"

# mostra as primeiras linhas como texto, para ver os metadados
with gzip.open(ARQ, "rt", encoding="utf-8") as f:
    for n in range(8):
        print(f.readline().rstrip()[:200])

# procura a linha de cabeçalho e lê a tabela
with gzip.open(ARQ, "rt", encoding="utf-8") as f:
    for pular, linha in enumerate(f):
        if "object_id" in linha:
            break
df = pd.read_csv(ARQ, skiprows=pular)

print("\nlinhas:", len(df))
print("colunas:", list(df.columns))
print("isprimary:", df["isprimary"].value_counts().to_dict())
print("RA de", df["ra"].min(), "a", df["ra"].max())
print("Dec de", df["dec"].min(), "a", df["dec"].max())