import os, sys
os.environ['HF_HOME'] = '/autodl-tmp/hf'
os.environ['HF_ENDPOINT'] = "https://hf-mirror.com"

from datasets import load_dataset
from text2knowledge.strategy1 import extract_concepts
from text2knowledge.normalizer import Normalizer, NormArg
import json
import contextlib

datas = load_dataset("bigbio/biored")
model_name = 'mistral:latest'


def find_substring_indices(text, substring):
    indices = []
    index = text.find(substring)
    while index != -1:
        indices.append(index)
        index = text.find(substring, index + 1)
    return indices

def overlap(r1, r2):
    # print(r1, r2)
    if r1[1] >= r2[0] and r1[0] <= r2[1]:
        return True
    return False

def calc_score(extracted, gt, text, offset):
    # print(offset)
    # get occurance of extracted entities in text
    for entity in extracted:
        name = entity['entity']
        occur = [[i+offset, i+offset+len(name)] for i in find_substring_indices(text, name)]
        entity['offsets'] = occur
    COR = 0
    INC = 0
    MIS = 0
    SPU = 0
    for entity in gt:
        is_COR = False
        is_MIS = True
        loc = entity['offsets'][0]
        for e in extracted:
            for o in e['offsets']:
                if overlap(loc, o):
                    is_MIS = False
                    if e['DBID'] == entity['DBID']:
                        is_COR = True
        if is_MIS:
            MIS += 1
        elif is_COR:
            COR += 1
        else:
            INC += 1
    for entity in extracted:
        is_SPU = True
        locs = entity['offsets']
        for o in locs:
            for e in gt:
                if overlap(o, e['offsets'][0]):
                    is_SPU = False
        if is_SPU:
            SPU += 1
    POS = COR + INC + MIS
    ACT = COR + INC + SPU
    P = COR / ACT
    R = COR / POS
    F1 = 2 * P * R / (P + R) if P + R else 0
    return F1


disease_normalizer = Normalizer(NormArg("dmis-lab/biosyn-biobert-ncbi-disease", "/root/text2knowledge/data/dictionary/merged/dict_Disease.txt"))
chemical_normalizer = Normalizer(NormArg("dmis-lab/biosyn-sapbert-bc5cdr-chemical", "/root/text2knowledge/data/dictionary/merged/dict_ChemicalCompound.txt"))
gene_normalizer = Normalizer(NormArg("dmis-lab/biosyn-sapbert-bc2gn", "/root/text2knowledge/data/dictionary/merged/dict_Gene.txt"))

def normalize(name, category):
    if category in ['Disease', 'Symptom', 'DiseaseOrPhenotypicFeature']:
        model = disease_normalizer
    elif category in ['Compound', 'ChemicalEntity']:
        model = chemical_normalizer
    elif category in ['Gene', 'Protein', 'CellularComponent', 'GeneOrGeneProduct']:
        model = gene_normalizer
    else:
        raise NotImplementedError(f'Unknown category: {category}')
    result = model.normalize(name)
    return result['predictions'][0]['id']


scores = []
for data in datas['test']:
    pmid = data['pmid']
    print(pmid)
    passages = data['passages']
    entities = data['entities']
    abstract = passages[1]['text'][0]
    abstract_offset = passages[1]['offsets'][0][0]
    metadata = {
        'pmid': pmid,
        'type': 'abstract',
    }

    score = 0
    
    # with open(os.devnull, "w") as f, contextlib.redirect_stdout(f):
        # entities_extracted = extract_concepts(abstract, model=model_name, metadata=metadata)
    entities_extracted = extract_concepts(abstract, model=model_name, metadata=metadata)
    
    
    if entities_extracted:
        # from pprint import pprint
        # pprint([entity for entity in entities_extracted if ])
        # normalize extracted entities to DB
        entities_extracted_normalized = []
        for entity in entities_extracted:
            name = entity['entity']
            category = entity['category']
            if category not in ['Gene', 'CellularComponent', 'Compound', 'Disease', 'Symptom', 'Protein']:
                continue
            ID = normalize(name, category)
            entity['DBID'] = ID
            entities_extracted_normalized.append(entity)

        

        # normalize answer to DB as well in order to calc score later
        entities_ans_normalized = []
        for entity in entities:
            name = entity['text'][0]
            category = entity['semantic_type_id']
            if category not in ['ChemicalEntity', 'DiseaseOrPhenotypicFeature', 'GeneOrGeneProduct']:
                continue
            ID = normalize(name, category)
            entity['DBID'] = ID
            entities_ans_normalized.append(entity)

        score = calc_score(entities_extracted_normalized, entities_ans_normalized, abstract, abstract_offset)  # TODO
        print(f'F1: {score}')
        scores.append(score)

        # disease_entities_extracted = [e for e in entities_extracted if e['category'] == 'Disease']
        # print(disease_entities_extracted)
        # entities_str = json.dumps(entities_extracted, indent=4)
        # print(entities_str)
    
    else:
        print("No entities found.")

print(sum(scores) / len(scores))