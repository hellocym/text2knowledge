import pandas as pd
import os

def formatter(table, path):
    sep = '\t' if table.endswith('.tsv') else ','
    df = pd.read_csv(table, sep=sep)
    df['synonyms'].fillna("none", inplace=True)
    groups = df.groupby(df.label)
    for group in groups:
        temp_group = list(group)
        file_name = f"dict_{temp_group[0]}.txt"
        temp_group[1]["ftext"] =temp_group[1].apply(get_text, axis=1)
        formatted_text = '\n'.join(temp_group[1]['ftext'].astype(str).values)
        file_path = os.path.join(path, file_name)
        with open(file_path, 'w') as file:
            file.write(formatted_text)

def get_text(row):
    if row['synonyms'] == "none":
        return f"{row['id']}||{row['name']}"
    else:
        return f"{row['id']}||{row['name']}|{row['synonyms']}"

if __name__ == "__main__":
    formatter("~/text2knowledge/data/entities.tsv", "" )
