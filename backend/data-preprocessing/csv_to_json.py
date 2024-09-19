import sys
import csv
import argparse
import re
import json
import os
from collections import Counter, defaultdict
from datetime import datetime

def read_csv(file_path):
    data = []
    try:
        with open(file_path, 'r', newline='', encoding='utf-8') as csvfile:
            csv_reader = csv.DictReader(csvfile)
            for row in csv_reader:
                data.append(row)
        return data
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        sys.exit(1)
    except csv.Error as e:
        print(f"Error reading CSV file: {e}")
        sys.exit(1)

def parse_date(date_string):
    # Remove the timezone part (e.g., 'CET')
    date_string_without_tz = ' '.join(date_string.split()[:-2]) + ' ' + date_string.split()[-1]
    return datetime.strptime(date_string_without_tz, '%a %b %d %H:%M:%S %Y')


def format_date(date):
    return date.strftime('%Y-%m-%d')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Read a CSV file and filter features")
    parser.add_argument("-i", "--input", required=True, help="Path to the input CSV file")
    parser.add_argument("-o", "--output", required=True, help="Path to the output JSON file")
    parser.add_argument("--from", dest="from_date", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--to", dest="to_date", help="End date (YYYY-MM-DD)")
    parser.add_argument("-n", "--min-occurrences", type=int, default=1, help="Minimum number of occurrences of a feature to be included")
    
    args = parser.parse_args()

    csv_file_path = args.input
    output_file_path = args.output
    from_date = datetime.strptime(args.from_date, '%Y-%m-%d') if args.from_date else None
    to_date = datetime.strptime(args.to_date, '%Y-%m-%d') if args.to_date else None
    min_occurrences = args.min_occurrences

    csv_data = read_csv(csv_file_path)
    
    # Filter data based on date range
    filtered_data = []
    for row in csv_data:
        if 'Date' in row:
            review_date = parse_date(row['Date'])
            if (from_date is None or review_date >= from_date) and (to_date is None or review_date <= to_date):
                filtered_data.append(row)

    print(f"Successfully read {len(filtered_data)} rows from {csv_file_path} within the specified date range")

    # Create subfolder with date interval
    date_interval = f"{args.from_date or 'start'}_{args.to_date or 'end'}"
    output_folder = os.path.join(os.path.dirname(output_file_path), date_interval)
    os.makedirs(output_folder, exist_ok=True)

    # Update output_file_path to use the new subfolder
    output_file_path = os.path.join(output_folder, os.path.basename(output_file_path))

    # Process TransFeatExFeatures
    transfeatex_features = []
    for row in filtered_data:
        if 'TransFeatExFeatures' in row:
            features = row['TransFeatExFeatures'].split('; ')
            transfeatex_features.extend(features)

    features = [' '.join(re.findall(r'[A-Z][a-z]*', feature)) for feature in transfeatex_features if feature]
    features = [feature.lower() for feature in features]
    # Remove empty features
    features = [feature for feature in features if feature.strip()]

    # Sort features by frequency and limit to those with at least min_occurrences
    feature_counts = Counter(features)
    sorted_features = sorted(feature_counts.items(), key=lambda x: x[1], reverse=True)
    frequent_features = [feature for feature, count in sorted_features if count >= min_occurrences]

    print(f"\nNumber of features with at least {min_occurrences} mentions: {len(frequent_features)}")
    print("Sample of frequent features:")
    for feature in frequent_features[:20]:  # Print first 20 frequent features as a sample
        print(f"{feature}: {feature_counts[feature]} mentions")
   
    # Save frequent features as JSON array
    try:
        with open(output_file_path, 'w', encoding='utf-8') as jsonfile:
            json.dump([feature for feature in frequent_features], jsonfile, indent=4)
        print(f"\nSuccessfully saved {len(frequent_features)} features to {output_file_path}")
    except IOError as e:
        print(f"Error writing to JSON file: {e}")
        sys.exit(1)
        
    # Group features by ApplicationId
    app_features = defaultdict(list)
    for row in filtered_data:
        if 'TransFeatExFeatures' in row and 'ApplicationId' in row:
            app_id = row['ApplicationId']
            features = row['TransFeatExFeatures'].split('; ')
            app_features[app_id].extend(features)

    # Process and save features for each ApplicationId
    for app_id, features in app_features.items():
        # Process features
        processed_features = [' '.join(re.findall(r'[A-Z][a-z]*', feature)) for feature in features if feature]
        processed_features = [feature.lower() for feature in processed_features]
        processed_features = [feature for feature in processed_features if feature.strip()]

        # Count and sort features
        feature_counts = Counter(processed_features)
        sorted_features = sorted(feature_counts.items(), key=lambda x: x[1], reverse=True)
        frequent_features = [feature for feature, count in sorted_features if count >= min_occurrences]

        # Create output file name
        app_output_file = os.path.join(output_folder, f"{os.path.splitext(os.path.basename(output_file_path))[0]}_{app_id}.json")

        # Save frequent features as JSON array
        try:
            with open(app_output_file, 'w', encoding='utf-8') as jsonfile:
                json.dump([feature for feature in frequent_features], jsonfile, indent=4)
            print(f"\nSuccessfully saved {len(frequent_features)} features for {app_id} to {app_output_file}")
        except IOError as e:
            print(f"Error writing to JSON file for {app_id}: {e}")
