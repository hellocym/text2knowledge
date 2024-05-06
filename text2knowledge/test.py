from datasets import load_dataset
from text2knowledge.strategy1 import extract_concepts
import json
import os, sys
import contextlib


datas = load_dataset("bigbio/biored")
model_name = 'mistral:latest'

scores = []
for data in datas['test']:
    pmid = data['pmid']
    passages = data['passages']
    entities = data['entities']
    abstract = passages[1]['text'][0]
    abstract_offset = passages[1]['offsets'][0]
    metadata = {
        'pmid': pmid,
        'type': 'abstract',
    }

    score = 0
    #with open(os.devnull, "w") as f, contextlib.redirect_stdout(f):
    entities_extracted = extract_concepts(abstract, model=model_name, metadata=metadata)
    
    
    if entities_extracted:
        # normalize extracted entities to DB
        entities_extracted_normalized = []
        for entity in entities_extracted:
            name = entity['entity']
            category = entity['category']
            ID = normalize(name, category) # TODO
            entity['DBID'] = ID
            entities_extracted_normalized.append(entity)

        # normalize answer to DB as well in order to calc score later
        entities_ans_normalized = []
        for entity in entities:
            name = entity['text'][0]
            category = entity['semantic_type_id']
            ID = normalize(name, category) # TODO
            entity['DBID'] = ID
            entities_ans_normalized.append(entity)

        score = calc_score(entities_extracted_normalized, entities_ans_normalized)  # TODO
        scores.append(score)

        # disease_entities_extracted = [e for e in entities_extracted if e['category'] == 'Disease']
        # print(disease_entities_extracted)
        # entities_str = json.dumps(entities_extracted, indent=4)
        # print(entities_str)
    
    else:
        print("No entities found.")


