from sentence_transformers import SentenceTransformer
from sentence_transformers import CrossEncoder
from Levenshtein import distance as lev
from functools import lru_cache
import faiss
from sklearn.preprocessing import normalize
import numpy as np
model_path = 'BAAI/bge-base-en-v1.5'
reranker_model = CrossEncoder("BAAI/bge-reranker-base", max_length=512)


def get_lev_distance(key1,key2):
    return lev(key1,key2)

def get_exact_match(mappings,key1):
    if key1 in mappings.keys():
        return mappings[key1]
    else:
        return False
    
def get_min_distance_product(mappings,key1):
    min_distance = 100
    index = None
    for key2 in mappings.keys():
        dist = get_lev_distance(key1,key2)
        if not index:
            index = key2
            min_distance = dist
        elif dist < min_distance:
            index = key2
            min_distance = dist
    return min_distance, index






@lru_cache()
def load_model():
    model = SentenceTransformer(model_name_or_path=model_path,device='cpu')
    return model

model = load_model()
    
def get_induvidual_products(products_df):
    required_columns_df = products_df[['sku','short_name','name','parent_sku','status','category','sub_category','is_family_head','is_individual','product_code','is_deleted']]
    induvidual_products = required_columns_df[((required_columns_df['is_family_head']==False) & (required_columns_df['status']=='Active'))]
    induvidual_products.to_csv('induvidual_prpducts.csv')
    return induvidual_products



def save_index(index,index_path):
    faiss.write_index(index, index_path)

def load_index(index_file_path):
    index_loaded = faiss.read_index(index_file_path)
    return index_loaded

def generate_embeddings(products_list):
    model = load_model()
    product_embeddings = model.encode(products_list)
    normalized_embeddings = normalize(product_embeddings)
    embedding_dim = normalized_embeddings.shape[1]
    index = faiss.IndexFlatIP(embedding_dim)
    index.add(normalized_embeddings)
    return index, normalized_embeddings
    

def generate_product_embeddings(products_df):
    products_list = list(products_df['name'])
    index, normalized_embeddings = generate_embeddings(products_list)
    return index, normalized_embeddings

def filter_indexes(indexes, dists, dist_threshold = 0.6):
    new_indexes = []
    new_dists = []
    start = True
    for idx, dist in zip(indexes[0],dists[0]):
        if start:
            previous_dist=dist
            start = False
        if dist>dist_threshold and previous_dist-dist < 0.05:
            new_indexes.append(idx)
            new_dists.append(dist)
            previous_dist = dist
    return [new_indexes], [new_dists]


def get_top_matches(product_text,index,product_list, dist_threshold = 0.5):
    product_text_embedding = model.encode([product_text])[0]
    normalized_product_text_embedding = normalize([product_text_embedding])[0]
    n=10
    dists,indexes = index.search(np.array([normalized_product_text_embedding]),n)
    indexes, dists = filter_indexes(indexes,dists, dist_threshold=dist_threshold)
    filtered_products = []
    for idx in indexes[0]:
        filtered_products.append(product_list[idx])
    return filtered_products,indexes

def get_best_match_with_reranker(product_text,filtered_products):
    score_list = []
    i = 0
    max_score = 0
    max_score_index = 0
    for product in filtered_products:
        score = reranker_model.predict([(product_text,product)])
        score_list.append(score)
        if score > max_score:
            max_score_index = i
            max_score = score
        i = i + 1
    return max_score_index




def get_matched_product(product_text, products_catalog_names_list,product_sku_list,index):
    filtered_products,indexes = get_top_matches(product_text,index,products_catalog_names_list)
    print(filtered_products)
    if not indexes[0]:
        product_sku = None
        filtered_products = [None]
    else:
        min_score_index = get_best_match_with_reranker(product_text,filtered_products)
        product_sku = product_sku_list[indexes[0][min_score_index]]
    return filtered_products[min_score_index],product_sku


