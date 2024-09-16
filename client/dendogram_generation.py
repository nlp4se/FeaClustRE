import json
import requests
import argparse
import logging
import os

# Set up logging
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
    url = 'http://127.0.0.1:3008/dendogram/generate?affinity=bert-embedding-cosine'
    headers = {'Content-Type': 'application/json'}
    app_name = input_file.split(os.path.sep)[-1].removeprefix('features_').removesuffix('.json')
    
    logging.info(f"Preparing request for app: {app_name}")

    # Send the request
    try:
        response = requests.post(url, headers=headers, json={'app_name': app_name, 'features': data})
        logging.info(f"Request sent to {url}")
    except requests.RequestException as e:
        logging.error(f"Error sending request: {e}")
        return

    # Check if the request was successful
    if response.status_code == 200:
        logging.info("Dendogram generated successfully.")
        # You can add more code here to handle the response if needed
    else:
        logging.error(f"Error generating dendogram. Status code: {response.status_code}")
        logging.error(f"Response: {response.text}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a dendogram from a JSON file.")
    parser.add_argument("input_file", help="Path to the input JSON file")
    args = parser.parse_args()

    logging.info("Starting script execution")
    generate_dendogram(args.input_file)
    logging.info("Script execution completed")
