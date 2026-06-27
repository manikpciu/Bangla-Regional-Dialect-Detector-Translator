import pandas as pd
import json
import os

# Load CSV
df = pd.read_csv("dataset/dialect_translation.csv")

# Create translations folder
os.makedirs("translations", exist_ok=True)

# Get unique regions
regions = df["Region"].unique()

for region in regions:

    region_df = df[df["Region"] == region]

    data = {}

    for _, row in region_df.iterrows():

        data[str(row["Text"]).strip()] = str(row["Standard_Bangla"]).strip()

    filename = f"translations/{region.lower()}.json"

    with open(filename, "w", encoding="utf-8") as f:
        json.dump(
            data,
            f,
            ensure_ascii=False,
            indent=4
        )

print("✅ All JSON files created successfully!")