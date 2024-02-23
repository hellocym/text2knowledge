from sklearn.metrics.pairwise import cosine_similarity

import torch
from transformers import BertTokenizer, BertModel
import argparse
import logging

import torch

from transformers import BertConfig, BertForPreTraining, load_tf_weights_in_bert


class BioBERT:
    def __init__(self, model_name, threshold):
        do_lower_case = True
        self.model = BertModel.from_pretrained(model_name)
        self.tokenizer = BertTokenizer.from_pretrained(model_name, do_lower_case=do_lower_case)
        self.threshold = threshold

    def embed_text(self, text):
        input_ids = torch.tensor(self.tokenizer.encode(text)).unsqueeze(0)  # Batch size 1
        outputs = self.model(input_ids)
        last_hidden_states = outputs[0]  # The last hidden-state is the first element of the output tuple
        return last_hidden_states

    def get_similarity(self, em, em2):
        return cosine_similarity(em.detach().numpy(), em2.detach().numpy())
    
    def similarity_threshold(self, term1: str, term2: str)->bool:
        em1 = self.embed_text(term1).mean(1)
        em2 = self.embed_text(term2).mean(1)
        similarity = self.get_similarity(em1, em2)
        print(similarity)
        return True if similarity >= self.threshold else False