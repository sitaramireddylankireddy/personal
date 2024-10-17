import requests, json, re
from src.config import API_KEY
# Define your API key

# Define the endpoint and headers
url = "https://api.openai.com/v1/chat/completions"
headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}

user_products_message = "sweatshirt of size M 2pieces tomato 4kg onion 5kg"
max_tokens = 4096

def extract_json_from_gpt_response(response):
    # Use a regex to find a JSON object in the response
    json_pattern = r'({.*?})'
    match = re.search(json_pattern, response, re.DOTALL)

    if match:
        json_str = match.group(1)
        try:
            # Attempt to parse the extracted JSON
            parsed_json = json.loads(json_str)
            return parsed_json
        except json.JSONDecodeError:
            print("Error: The extracted text is not valid JSON.")
            return None
    else:
        print("Error: No JSON object found in the response.")
        return None

def generate_prompt(user_products_message):
    prompt = """Below is the text contains products and its quantities please give the output only in json format with product as key and quantity, unit of measure(uom) as value dict.Please provide only the JSON data without any additional text. 
    #########
    Tasks: 1. Extract product  and its quantity, unit of measure(uom) for each product from the text.
           2. give output in json format with product as key and quantity, unit of measure(uom) as value dictionary for product. Value dictionary should contain quantity and uom as keys and their respective values.
           3. Check weather the output is valid json or not, if not valid json correct it. 
    #########
    output format: strictly in valid json format.
    ##########
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

