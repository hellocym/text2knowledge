import transformers_embedder as tre
import pandas as pd
import torch
import numpy as np

import gc

import faiss


d = 768
index = faiss.IndexFlatIP(d)


tokenizer = tre.Tokenizer("microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext")

model = tre.TransformersEmbedder(
    "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext", subword_pooling_strategy="sparse", layer_pooling_strategy="mean"
)


file_path = 'entities.tsv'
df = pd.read_csv(file_path, sep='\t')

names = df.loc[:,'name'].to_list()


chunk_len = 512
last = 0

device = 'cuda' if torch.cuda.is_available() else 'cpu'
# tokenizer = tokenizer.to(device)
model = model.to(device)

while last<=len(names):
    try:
        print(f'Processing ({last}-{last+chunk_len})')
        inputs = tokenizer(names[last:last+chunk_len], return_tensors=True, padding=True).to(device)
        outputs = model(**inputs)
    except Exception as e:
        print(e)
        chunk_len //= 2
        print(f'chunk lengh reduced to {chunk_len}')
        continue
    
    embs = outputs.word_embeddings
    sent_vecs = [(embs[i][:inputs['sentence_lengths'][i]].sum(dim=0) / inputs['sentence_lengths'][i]).detach().cpu().numpy() for i in range(embs.shape[0])]
    vecs = np.array(sent_vecs)
    faiss.normalize_L2(vecs)
    index.add(vecs)
    faiss.write_index(index, "biovecs.index")
    # print(outputs.word_embeddings.shape)
    del inputs
    del outputs.word_embeddings
    del outputs
    torch.cuda.empty_cache()
    gc.collect()
    last += chunk_len
    chunk_len += 128
    
    # index = faiss.read_index("biovecs.index")

        
