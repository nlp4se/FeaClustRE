import json
import csv
import os
import argparse
from collections import defaultdict


def parse_reviews(input_file):
    # Read the JSON file
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Extract reviews and add additional fields
    reviews_data = []
    grouped_reviews = defaultdict(list)

    for app in data:
        app_name = app.get('app_name')
        app_package = app.get('reviews')[0].get('package') if app.get('reviews') else None
        app_categoryId = app.get('categoryId')

        for review in app.get('reviews', []):
            review_row = {
                'app_name': app_name,
                'app_package': app_package,
                'app_categoryId': app_categoryId,
                'reviewId': review.get('reviewId'),
                'review': review.get('review'),
                'reply': review.get('reply'),
                'userName': review.get('userName'),
                'score': review.get('score'),
                'at': review.get('at')
            }
            reviews_data.append(review_row)
            grouped_reviews[app_categoryId].append(review_row)

    return reviews_data, grouped_reviews


def write_csv(output_file, data, fieldnames):
    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames, escapechar='\\', quoting=csv.QUOTE_MINIMAL)
        writer.writeheader()
        writer.writerows(data)


def process_folder(input_folder, output_folder):
    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # A dictionary to group reviews by categoryId across all files
    combined_grouped_reviews = defaultdict(list)

    # Iterate over all JSON files in the input folder
    for file_name in os.listdir(input_folder):
        if file_name.endswith('.json'):
            input_file = os.path.join(input_folder, file_name)
            app_output_file = os.path.join(output_folder, file_name.replace('.json', '.csv'))
            
            print(f"Processing {input_file} -> {app_output_file}")
            reviews_data, grouped_reviews = parse_reviews(input_file)

            # Write CSV for each app
            if reviews_data:
                fieldnames = list(reviews_data[0].keys())
                write_csv(app_output_file, reviews_data, fieldnames)

            # Merge reviews into combined grouped reviews
            for categoryId, reviews in grouped_reviews.items():
                combined_grouped_reviews[categoryId].extend(reviews)

    # Write separate CSV files for each categoryId
    for categoryId, reviews in combined_grouped_reviews.items():
        category_output_file = os.path.join(output_folder, f"{categoryId}.csv")
        print(f"Saving reviews for categoryId={categoryId} -> {category_output_file}")

        # Write CSV for each categoryId
        if reviews:
            fieldnames = list(reviews[0].keys())
            write_csv(category_output_file, reviews, fieldnames)

    print(f"All reviews processed. App files and category files saved in {output_folder}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parse JSON files and save reviews for apps and categories")
    parser.add_argument('--input_folder', required=True, help='Path to the input folder containing JSON files')
    parser.add_argument('--output_folder', required=True, help='Path to the output folder to save CSV files')

    args = parser.parse_args()
    process_folder(args.input_folder, args.output_folder)
