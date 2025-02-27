import requests
import json
import subprocess

base_url = "http://127.0.0.1"
port = "3008"
endpoint = "dendogram/generate"

thresholds = [0.1, 0.2, 0.3, 0.4, 0.5]
verb_weights = [0.1, 0.25, 0.331, 0.5, 0.75, 0.9]
obj_weights = [0.9, 0.75, 0.669, 0.5, 0.25, 0.1]
affinity_models = ['bert', 'tf-idf', 'paraphrase']

json_file_path = "small_whatsapp_example.json"
with open(json_file_path, 'r') as json_file:
    json_data = json.load(json_file)

def construct_url(base_url, port, endpoint, params):
    if port:
        full_url = f"{base_url}:{port}/{endpoint}"
    else:
        full_url = f"{base_url}/{endpoint}"
    return (f"{full_url}?preprocessing=true&affinity={params['affinity']}&metric=cosine&threshold={params['threshold']}"
            f"&linkage=average&verb-weight={params['verb_weight']}&obj-weight={params['obj_weight']}")

def main():
    for affinity_model in affinity_models:
        for threshold in thresholds:
            for verb_weight, obj_weight in zip(verb_weights, obj_weights):
                params = {
                    "affinity": affinity_model,
                    "threshold": threshold,
                    "verb_weight": verb_weight,
                    "obj_weight": obj_weight
                }

                url = construct_url(base_url, port, endpoint, params)

                try:
                    response = requests.post(url, json=json_data)
                    if response.status_code == 200:
                        print(f"Success: {response.status_code} - {url}")
                    else:
                        print(f"Error: {response.status_code} - {url}")
                except requests.exceptions.RequestException as e:
                    print(f"Request failed: {e}")

    try:
        subprocess.run(['python', 'visualizator.py'], check=True)
        print("Visualizator script executed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Error executing visualizator.py: {e}")
    except FileNotFoundError:
        print("visualizator.py not found. Make sure it's in the same directory as this script.")

if __name__ == "__main__":
    main()
