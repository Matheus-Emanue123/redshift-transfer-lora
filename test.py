import pyvo

tap = pyvo.dal.TAPService("https://irsa.ipac.caltech.edu/TAP")

tabelas = tap.search("SELECT table_name, description FROM TAP_SCHEMA.tables").to_table().to_pandas()
cosmos = tabelas[tabelas["table_name"].str.contains("cosmos", case=False)]
print(cosmos.to_string())

for nome in cosmos["table_name"]:
    cols = tap.search(f"SELECT column_name FROM TAP_SCHEMA.columns WHERE table_name = '{nome}'").to_table()
    print(nome, len(cols), "colunas")
    print(list(cols["column_name"]))