# Use of Faiss to setup local RAG.
# install sentence-transformers faiss-cpu torch>=2.6 torchvision

import os
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer


'''
df = 
詳細
-----
あああ,
いいい,
'''

def chunk_text(text:str, chuck_size=800, overlap=120):
    if not text:
        return []
    tokens = text.split()
    chunks = []
    i=0
    while i < len(tokens):
        chunk = " ".join(tokens[i:i+chunk_size])
        if chunk.strip()
            chunks.append(chunk)
        i += max(1, chunk_size-overlap)
    return chunks

rows = df.select('詳細').toLocalIterator()
records = []
for r_idx, r in enumerate(rows):
    text = r['詳細']
    chunks = chunk_text(text, chunk_size=800, overlap=120)
    for c_idx, ch in enumerate(chunks):
        records.append({
        "row_id":r_idx,
        "chunk_id":c_idx,
        "chunk_text":ch
        })

model_name = "embaas/sentence-transformers-multilingual-e5-base"
model = SentenceTransformer(model_name)

def embed_passages(texts):
    prefixed = ['passage: ' + t for t in texts]
    return model.encode(prefixed, normalized_embeddings=True) #Normalize the vec.

# Embed
chunk_texts = [rec['chunk_text'] for rec in records]
emb = embed_passages(chunk_texts).astype('float32') #Faiss expects float32

# Store vecs to Faiss index.
embed_dim = emb.shape[1]
index = faiss.IndexFlat(embed_dim)
index.add(emb)

index_dir = './vector_index'
index_path = os.path.join(index_dir, 'vecorDB.index')

faiss.write_index(index, index_path)

#Semantic search
def search(query: str, top_k=5):
    q = model.encode(['query :' + query], normalize_embeddings=True).astype('float32')
    idx = faiss.read_index(index_path)
    scores, ids = idx.search(q, top_k)
    hits = []
    for rank in range(top_k):
        rid = int(ids[0, rank])
        rec = records[rid]
        hits.append({
            'rank':rank+1,
            'score':float(scores[0,rank]),
            'row_id':rec['row_id'],
            'chunk_id':rec['chunk_id'],
            'chunk_text':rec['chunk_text'][:200] + ('...' if len(rec['chunk_text']>200 else  '')
            })
    return hits

#query it.
query = "てすと、てすと"
res = search(query, top_k=5)
