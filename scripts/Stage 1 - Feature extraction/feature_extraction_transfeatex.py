import os
import json
import argparse
import requests

def process_reviews(input_folder, output_folder, url):
    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)

    for file_name in os.listdir(input_folder):
        if file_name.endswith(".json"):
            input_file_path = os.path.join(input_folder, file_name)

            # Read the input JSON file
            with open(input_file_path, 'r', encoding='utf-8') as file:
                try:
                    app_data = json.load(file)
                except json.JSONDecodeError:
                    print(f"Error decoding JSON in file: {file_name}")
                    continue

            # print(app_data)
            
            # Send POST request
            headers = {'Content-Type': 'application/json'}
            try:
                response = requests.post(url, headers=headers, json=app_data)
                response.raise_for_status()
            except requests.exceptions.RequestException as e:
                print(f"Error sending request for file {file_name}: {e}")
                continue

            # Save the response
            output_file_path = os.path.join(output_folder, file_name)
            with open(output_file_path, 'w', encoding='utf-8') as output_file:
                output_file.write(response.text)

            print(f"Processed {file_name} and saved response to {output_file_path}")

if __name__ == "__main__":
    # Argument parsing
    parser = argparse.ArgumentParser(description="Process app reviews and send POST requests.")
    parser.add_argument("--input_folder", required=True, help="Path to the folder containing input JSON files.")
    parser.add_argument("--output_folder", required=True, help="Path to the folder to save the responses.")
    parser.add_argument("--url", required=True, help="URL to send the POST requests.")
    
    args = parser.parse_args()
    
    # Process reviews
    process_reviews(args.input_folder, args.output_folder, args.url)
