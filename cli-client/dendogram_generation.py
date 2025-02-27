import json
import requests
import argparse
import logging
import os

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def generate_dendogram(input_file):
    logging.info(f"Starting dendogram generation for file: {input_file}")

    # Read the JSON file
    try:
        with open(input_file, 'r') as file:
            data = json.load(file)
        logging.info(f"Successfully loaded JSON data from {input_file}")
    except Exception as e:
        logging.error(f"Error reading JSON file: {e}")
        return

    # Prepare the request
    url = f'http://127.0.0.1:3008/dendogram/generate?affinity={args.affinity}&linkage={args.linkage}&threshold={args.threshold}&verb_weight={args.verb_weight}&object_weight={args.object_weight}'
    headers = {'Content-Type': 'application/json'}
    app_name = input_file.split(os.path.sep)[-1].removeprefix('features_').removesuffix('.json')
    
    logging.info(f"Preparing request for app: {app_name}")

    try:
        response = requests.post(url, headers=headers, json={'app_name': app_name, 'features': data})
        logging.info(f"Request sent to {url}")
    except requests.RequestException as e:
        logging.error(f"Error sending request: {e}")
        return

    if response.status_code == 200:
        logging.info("Dendogram generated successfully.")
    else:
        logging.error(f"Error generating dendogram. Status code: {response.status_code}")
        logging.error(f"Response: {response.text}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a dendogram from a JSON file.")
    parser.add_argument("input_file", help="Path to the input JSON file")
    parser.add_argument("affinity", help="Affinity (e.g.; bert-embedding-cosine)")
    parser.add_argument("linkage", help="Linkage for agglomerative algorithm: ward, complete, average, single")
    parser.add_argument("threshold", help="Distance threshold for clustering")
    parser.add_argument("verb_weight", help="Weight for verb features")
    parser.add_argument("object_weight", help="Weight for object features")
    args = parser.parse_args()

    logging.info("Starting script execution")
    generate_dendogram(args.input_file)
    logging.info("Script execution completed")
