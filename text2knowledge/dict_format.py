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
    
    
def merge(rd, dd):
    merged = ""

    with open(rd, "r") as f1, open(dd, "r") as f2:
        lines1 = f1.read().split("\n")
        lines2 = f2.read().split("\n")
        data1 = {l.split("||")[0]:l.split("||")[1].split("|") for l in lines1 if l}
        data2 = {l.split("||")[0]:l.split("||")[1].split("|") for l in lines2 if l}
        for key, value_list in data2.items():
            if key in data1:
                data1[key] = list(set(data1[key] + value_list))
            else:
                data1[key] = value_list
        for k, v in data1.items():
            line = f"{k}||" + "|".join(v) + "\n"
            merged += line

    with open(rd, "w") as f:
        f.write(merged)
