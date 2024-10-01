from src.indexing import get_top_matches, generate_embeddings

product_list = ['iphone 4GB RAM', 'iphone','iphone 8GB RAM']

index, normalized_embeddings = generate_embeddings(product_list)
product = 'iphone with 4GB RAM'
filtered_products = get_top_matches(product,index,product_list)
print(filtered_products)
print(normalized_embeddings)

