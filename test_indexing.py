
# message -> gpt -> {products names and its qauantities}->
# Catalog -> embeddings
# product name-> product embedding 
# search in index file using product embeddings for best match 
# wrong
# sweat shirt with size M-> shirt
#'sweat shirt with size M': 'Long Sleeve Sweatshirt with letter / grey · M'
# "egg": 'tomato'
# 
#dictionary with mappings-> embeddings on keys
# product name-> embeddings
# modify user product name

# output as sku, quantity
# saving index file
# gpt integration




from src.indexing import get_matched_product, get_induvidual_products
from src.genrate_product_details import chat_with_gpt
from src.indexing import load_index
import pandas as pd

# use generate_product_embeddings function to generate index and the use save_index to save embeddings file
# index = load_index('products_catalog.index')
message_from_customer = """Denim shoes black size 6.5 1 pair, Lago breifcase 1, casual fixed belt 110 cm 1 """
index_path = 'products_catalog.index'
products_df = pd.read_csv('data-1727686302676.csv')
embeded_products_catlog = get_induvidual_products(products_df)
products_catalog_names_list = list(embeded_products_catlog['name']) 
product_sku_list = list(embeded_products_catlog['sku'])
index = load_index(index_path)

products_json = chat_with_gpt(message_from_customer)
print('#####################products_json###############')
print(products_json)
for product_text in products_json.keys():
    product_name,product_sku = get_matched_product(product_text, products_catalog_names_list,product_sku_list,index)
    print('################product_details#############')
    print(product_name)
    print(product_sku)

