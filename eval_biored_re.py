import os, sys
# os.environ['HF_HOME'] = '/autodl-tmp/hf'
os.environ['HF_ENDPOINT'] = "https://hf-mirror.com"

from datasets import load_dataset
from text2knowledge.strategy1 import extract_concepts
from text2knowledge.normalizer import Normalizer, NormArg
import json
import contextlib
import torch

datas = load_dataset("bigbio/biored")
use_cuda = torch.cuda.is_available()


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

def calc_score(extracted, gt, task='type'):
    # print(offset)
    # get occurance of extracted entities in text
    # TP: extracted entity in gt,
    # FP: extracted entity not in gt,
    # FN: gt entity not in extracted
    TP = 0
    FP = 0
    FN = 0

    for relation in extracted:
        # print(relation)
        source_id = relation['source_id']
        target_id = relation['target_id']
        relation_type = relation['relation_type']

        # find the corresponding gt entity
        gt_entity = [e for e in gt if e['source_id'] == source_id and e['target_id'] == target_id and e['relation_type'] == relation_type]
        if gt_entity:
            TP += 1
        else:
            FP += 1
    for relation in gt:
        source_id = relation['source_id']
        target_id = relation['target_id']
        relation_type = relation['relation_type']
        # find the corresponding gt entity
        extracted_entity = [e for e in extracted if e['source_id'] == source_id and e['target_id'] == target_id and e['relation_type'] == relation_type]
        if extracted_entity:
            pass
        else:
            FN += 1

    print(f'TP: {TP}, FP: {FP}, FN: {FN}')
    # P = TP / (TP + FP) if TP + FP else 0
    # R = TP / (TP + FN) if TP + FN else 0
    # F1 = 2 * P * R / (P + R) if P + R else 0
    return TP, FP, FN

disease_normalizer = Normalizer(NormArg("dmis-lab/biosyn-biobert-ncbi-disease", "./data/dictionary/merged/dict_Disease.txt", use_cuda=use_cuda))
chemical_normalizer = Normalizer(NormArg("dmis-lab/biosyn-sapbert-bc5cdr-chemical", "./data/dictionary/merged/dict_Compound.txt", use_cuda=use_cuda))
gene_normalizer = Normalizer(NormArg("dmis-lab/biosyn-sapbert-bc2gn", "./data/dictionary/merged/dict_Gene.txt", use_cuda=use_cuda))
symptom_normalizer = Normalizer(NormArg("dmis-lab/biosyn-biobert-ncbi-disease", "./data/dictionary/merged/dict_Symptom.txt", use_cuda=use_cuda))


def normalize(name, category):
    if category in ['Disease', 'DiseaseOrPhenotypicFeature', 'OrganismTaxon']:
        model = disease_normalizer
    elif category in ['Symptom']:
        model = symptom_normalizer
    elif category in ['Compound', 'ChemicalEntity']:
        model = chemical_normalizer
    elif category in ['Gene', 'Protein', 'CellularComponent', 'GeneOrGeneProduct', 'CellLine', 'SequenceVariant']:
        model = gene_normalizer
    else:
        raise NotImplementedError(f'Unknown category: {category}')
    result = model.normalize(name)
    return result['predictions'][0]['id']

TP = 0
FP = 0
FN = 0

# scores = []
for i, data in enumerate(datas['test']):
    pmid = data['pmid']
    print('PMID: ', pmid)
    passages = data['passages']
    entities = data['entities']
    relations = data['relations']
    abstract = passages[1]['text'][0]
    abstract_offset = passages[1]['offsets'][0][0]
    metadata = {
        'pmid': pmid,
        'type': 'abstract',
    }

    score = 0
    
    relations_extracted = json.load(open(f'./extracted/biored/test_{i}.json', 'r'))
    # print(entities_extracted)
    # continue
    
    if relations_extracted:
        
        # normalize extracted entities to DB
        relations_extracted_normalized = []
        for relation in relations_extracted:
            source_name = relation['source_name']
            source_type = relation['source_type']
            target_name = relation['target_name']
            target_type = relation['target_type']
            source_id = normalize(source_name, source_type)
            target_id = normalize(target_name, target_type)

            # print(source_name, source_type, source_id)
            # print(target_name, target_type, target_id)
            relation['source_id'] = source_id
            relation['target_id'] = target_id
            
            relations_extracted_normalized.append(relation)
        print(relations_extracted_normalized)

        # print(relations_extracted_normalized)
        # continue

        # normalize answer to DB as well in order to calc score later
        relations_ans_normalized = []
        for relation_ans in relations:
            source_bio_id = relation_ans['concept_1']
            target_bio_id = relation_ans['concept_2']
            # get entity name from entities
            source_name = [e['text'][0] for e in entities if e['concept_id'] == source_bio_id][0]
            target_name = [e['text'][0] for e in entities if e['concept_id'] == target_bio_id][0]
            source_type = [e['semantic_type_id'] for e in entities if e['concept_id'] == source_bio_id][0]
            target_type = [e['semantic_type_id'] for e in entities if e['concept_id'] == target_bio_id][0]
            source_id = normalize(source_name, source_type)
            target_id = normalize(target_name, target_type)
            relation_ans['source_id'] = source_id
            relation_ans['target_id'] = target_id

            relation_ans['relation_type'] = relation_ans['type']
            relations_ans_normalized.append(relation_ans)
        print(relations_ans_normalized)
        exit()
        tp, fp, fn = calc_score(relations_extracted_normalized, relations_ans_normalized)
        TP += tp
        FP += fp
        FN += fn
        print(f'Current TP: {TP}, FP: {FP}, FN: {FN}')


        # disease_entities_extracted = [e for e in entities_extracted if e['category'] == 'Disease']
        # print(disease_entities_extracted)
        # entities_str = json.dumps(entities_extracted, indent=4)
        # print(entities_str)
    
    else:
        print("No entities found.")

print('TP: ', TP)
print('FP: ', FP)
print('FN: ', FN)
print(f'P: {TP / (TP + FP)}')
print(f'R: {TP / (TP + FN)}')
print(f'F1: {2 * TP / (2 * TP + FP + FN)}')
