from openai import OpenAI
import json
import numpy as np
import pathlib
import hashlib

PATH = pathlib.Path().resolve() / 'Scraper-files' / 'product_data' / 'prod_json'

client = OpenAI()

with open(PATH / 'embeddings.json', 'r') as file:
    embeddings = json.load(file)
with open(PATH / 'products.json', 'r') as file:
    products = json.load(file)

product_keys = list(products.keys())

MAX_PRODUCTS = 60
i = 0
for key in product_keys:
    if (i < MAX_PRODUCTS) and (products[key]['Supplier'] == 'Axor') and not (key in embeddings):
        entry = products[key]
        response = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[
                {
                    "role": "system", 
                    "content": "Olet tuotekuvauskone. Käytä kuvaa ja tietoja kuvaillaksesi tuotetta suomeksi lyhyesti. Vastaus ei saa olla liian pitkä. Kuvaile erityisesti ulkonäköä, muotoa sekä väriä. Käytä erityisesti kuvaa hyväksesi. Vältä sellaisia termejä kuten moderni, tyylikäs, luksus. Vältä mielipiteitä, vaan pidättäydy faktoissa." 
                },
                {
                    "role": "user",
                    "content": [
                        { 
                            "type": "image_url", 
                            "image_url": { "url": products[key]['ImageURLs'] } 
                        },
                        { 
                            "type": "text", 
                            "text": str(products[key]) 
                        }
                  ]
                },
            ],
            temperature=0.3
        )
        entry['ai_kuvaus'] = response.choices[0].message.content
        entry['ai_embedding'] = []
        embeddings[key] = entry
        
        i += 1
        print(i)

with open(PATH / 'embeddings.json', 'w') as file:
    json.dump(embeddings, file, indent=4)

def cosine_similarity(vector1, vector2):
    return np.dot(vector1, vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2))

#response = client.responses.create(
#    model="gpt-4.1-nano-2025-04-14",
#    input="Write a haiku about a game of chess."
#)
#
#print(response.output_text)