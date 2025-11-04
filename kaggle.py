import os
import zipfile
import pandas as pd

dataset = "osmi/mental-health-in-tech-survey"
output_dir = "mental_health_data"
max_rows = 2000 

os.environ['KAGGLE_CONFIG_DIR'] = os.path.expanduser("~/.kaggle")

os.system(f"kaggle datasets download -d {dataset} -p {output_dir} --force")

zip_path = os.path.join(output_dir, dataset.split("/")[-1] + ".zip")

with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(output_dir)

csv_files = [f for f in os.listdir(output_dir) if f.endswith(".csv")]
csv_path = os.path.join(output_dir, csv_files[0])

df = pd.read_csv(csv_path)

df = df.head(max_rows)

# 🔹 Guardar CSV final
df.to_csv("mental_health_tech_2000.csv", index=False, encoding="utf-8")
print(f"CSV generado con {len(df)} registros: mental_health_tech_2000.csv")