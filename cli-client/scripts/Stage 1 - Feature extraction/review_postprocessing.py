import os
import json
import pandas as pd
from argparse import ArgumentParser

def parse_and_add_column(csv_folder, json_folder, output_folder):
    # Ensure output folder exists
    os.makedirs(output_folder, exist_ok=True)
    
    # List all CSV and JSON files
    csv_files = {os.path.splitext(f)[0]: f for f in os.listdir(csv_folder) if f.endswith('.model_embeddings')}
    json_files = {os.path.splitext(f)[0]: f for f in os.listdir(json_folder) if f.endswith('.json')}
    
    for base_name, csv_file in csv_files.items():
        json_file = json_files.get(base_name)
        
        if not json_file:
            print(f"No matching JSON file found for {csv_file}. Skipping.")
            continue
        
        # Load the CSV file
        csv_path = os.path.join(csv_folder, csv_file)
        df = pd.read_csv(csv_path)
        
        # Rename 'extracted_features' to 'extracted_features_T-FREX'
        if 'extracted_features' in df.columns:
            df.rename(columns={'extracted_features': 'extracted_features_T-FREX'}, inplace=True)
        
        # Load the JSON file
        json_path = os.path.join(json_folder, json_file)
        with open(json_path, 'r') as jf:
            json_data = json.load(jf)
        
        # Build a mapping of reviewId to features
        review_features_map = {}
        for review in json_data.get("analyzed_reviews", []):
            review_id = review.get("reviewId")
            if review_id:
                features = [
                    sentence.get("featureData", {}).get("feature", "")
                    for sentence in review.get("sentences", [])
                ]
                # Concatenate features separated by ';'
                review_features_map[review_id] = ";".join(filter(None, features))
        
        # Add the 'extracted_features_TransFeatEx' column
        df['extracted_features_TransFeatEx'] = df['reviewId'].map(review_features_map)
        # Filter out rows where extracted_features_TransFeatEx is empty
        df = df[df['extracted_features_TransFeatEx'].notna() & (df['extracted_features_TransFeatEx'] != '')]
        # Save the updated DataFrame to the output folder
        output_path = os.path.join(output_folder, csv_file)
        df.to_csv(output_path, sep=',', index=False)
        print(f"Processed {csv_file} and saved to {output_path}")

if __name__ == "__main__":
    parser = ArgumentParser(description="Parse CSV files and add a column with features extracted from JSON files.")
    parser.add_argument("--csv_folder", required=True, help="Path to the folder containing CSV files.")
    parser.add_argument("--json_folder", required=True, help="Path to the folder containing JSON files.")
    parser.add_argument("--output_folder", required=True, help="Path to the output folder for processed CSV files.")
    
    args = parser.parse_args()
    parse_and_add_column(args.csv_folder, args.json_folder, args.output_folder)
