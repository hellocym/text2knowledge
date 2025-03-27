from datasets import load_dataset
import json
import os, sys
import contextlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from text2knowledge.strategy1 import graph_prompt
from text2knowledge.type_definition import BIORED_ENTITY_TYPES, BIORED_RELATION_TYPES

def process_single_data(data, model_name, base_url, index):
    pmid = data['pmid']
    passages = data['passages']
    relations = data['relations']
    abstract = passages[1]['text'][0]
    abstract_offset = passages[1]['offsets'][0]
    metadata = {
        'pmid': pmid,
        'type': 'abstract',
    }
    
    relations_extracted = graph_prompt(
        abstract,
        model=model_name,
        metadata=metadata,
        base_url=base_url,
        entity_types=BIORED_ENTITY_TYPES,
        relation_types=BIORED_RELATION_TYPES
    )
    
    if relations_extracted:
        save_path = f'./extracted/biored/test_{index}.json'
        with open(save_path, 'w') as f:
            f.write(json.dumps(relations_extracted, indent=4))
        return f"Entities saved to {save_path}"
    return "No entities found."

# 主程序部分
datas = load_dataset("bigbio/biored")
model_name = 'qwen2.5-7b-instruct-1m'
base_url = 'http://192.168.31.58:1234/v1'

# 设置线程池大小
max_workers = 5  # 可以根据需要调整线程数

with ThreadPoolExecutor(max_workers=max_workers) as executor:
    # 创建任务列表
    future_to_index = {
        executor.submit(process_single_data, data, model_name, base_url, i): i 
        for i, data in enumerate(datas['test'])
    }
    
    # 处理完成的任务
    for future in as_completed(future_to_index):
        index = future_to_index[future]
        try:
            result = future.result()
            print(f"Task {index}: {result}")
        except Exception as e:
            print(f"Task {index} generated an exception: {str(e)}")
