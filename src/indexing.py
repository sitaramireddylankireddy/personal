from sentence_transformers import SentenceTransformer
import faiss
from sklearn.preprocessing import normalize
import numpy as np
model_path = 'BAAI/bge-base-en-v1.5'


model = SentenceTransformer(model_name_or_path=model_path,device='cpu')

def generate_embeddings(products_list):
    product_embeddings = model.encode(products_list)
    normalized_embeddings = normalize(product_embeddings)
    embedding_dim = normalized_embeddings.shape[1]
    index = faiss.IndexFlatIP(embedding_dim)
    index.add(normalized_embeddings)
    return index, normalized_embeddings

def filter_indexes(indexes, dists):
    new_indexes = []
    new_dists = []
    start = True
    for idx, dist in zip(indexes[0],dists[0]):
        if start:
            previous_dist=dist
            start = False
        if dist>0.6 and previous_dist-dist < 0.05:
            new_indexes.append(idx)
            new_dists.append(dist)
            previous_dist = dist
    return [new_indexes], [new_dists]


def get_top_matches(product_text,index,product_list):
    product_text_embedding = model.encode([product_text])[0]
    normalized_product_text_embedding = normalize([product_text_embedding])[0]
    n=1
    dists,indexes = index.search(np.array([normalized_product_text_embedding]),n)
    indexes, dists = filter_indexes(indexes,dists)
    filtered_products = []
    for idx in indexes[0]:
        filtered_products.append(product_list[idx])
    return filtered_products



