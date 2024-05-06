import torch
import transformers_embedder as tre
import argparse
import logging
import numpy as np
import pandas as pd

import faiss


class BioBERT:
    def __init__(self, model_name, index_path, db_path):
        do_lower_case = True
        print('Loading model...')
        self.model = tre.TransformersEmbedder(model_name, subword_pooling_strategy="sparse", layer_pooling_strategy="mean"
)
        self.tokenizer = tre.Tokenizer(model_name)
        print('Loading index...')
        self.index = faiss.read_index(index_path)
        self.entities = pd.read_csv(db_path, sep='\t')

    def embed_text(self, text):
        input_ids = self.tokenizer(text, return_tensors=True, padding=True).to(device)
        outputs = self.model(**input_ids)
        return outputs.word_embeddings

    def query(self, entities, top_k=5):
        inputs = self.tokenizer(entities, return_tensors=True, padding=True)
        outputs = self.model(**inputs)
        embs = outputs.word_embeddings
        q = (embs.sum(dim=1)/embs.shape[1]).detach().numpy()  # phrase embedding
        faiss.normalize_L2(q)
        D, I = self.index.search(q, top_k)  # D for simularity, I for index
        E = [self.entities.loc[:,'name'][idx] for idx in I]  # E for entity names
        return D, I, E
    
