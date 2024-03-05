import transformers_embedder as tre
import pandas as pd


tokenizer = tre.Tokenizer("microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext")

model = tre.TransformersEmbedder(
    "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext", subword_pooling_strategy="sparse", layer_pooling_strategy="mean"
)


file_path = 'entities.tsv'
df = pd.read_csv(file_path, sep='\t')

names = df.loc[:,'name'].to_list()
inputs = tokenizer(names[:50], return_tensors=True, padding=True) #5000
outputs = model(**inputs)

embs = outputs.word_embeddings
print(embs.shape)


