import requests, json

# Define your API key
API_KEY = 'use_secret_key'  # Replace with your actual API key

# Define the endpoint and headers
url = "https://api.openai.com/v1/chat/completions"
headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}

user_products_message = "sweatshirt of size M 2pieces tomato 4kg onion 5kg"
max_tokens = 4096

def generate_prompt(user_products_message):
    prompt = """Below is the text contains products and its quantities please give the output only in json format with product as key and quantity, unit of measure(uom) as value dict.Please provide only the JSON data without any additional text. 
    Example 1:
    Input: Meera shampoo 250 ml, sweatshirt with M size, tomato 4kg onion 2kg
    Output: {'Meera shampoo': {'quantity:'250','uom': 'ml'}, 'sweatshirt with M size': {'quantity:'1','uom': None}, 'tomato': {'quantity:'4','uom': 'Kg'}, 'onion' : {'quantity:'2','uom': 'Kg'} }""" + """\n Now get the products and quantities for below sentence:
    Input:""" + user_products_message + "\nOutput:"
    return prompt

def generate_response(prompt):

    payload = {
        'model': 'gpt-4o-mini',
        'messages': [
            {'role': 'user', 'content': prompt}
        ],
        'max_tokens': max_tokens,
        'temperature': 0,
        'top_p':1
    }
    response = requests.post(
        "https://api.openai.com/v1/chat/completions", headers=headers, json=payload
    )
    answer = response.json()["choices"][0]["message"]["content"]
    # answer = post_process_output(answer)
    gpt_output = json.loads(answer)
    print(f"gpt_output : {gpt_output}")
    return gpt_output

def chat_with_gpt(user_products_message):
    prompt = generate_prompt(user_products_message)
    gpt_output = generate_response(prompt)
    return gpt_output

