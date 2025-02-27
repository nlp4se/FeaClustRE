import os
import json
import argparse
import unicodedata
import re
from lingua import Language, LanguageDetectorBuilder

# Set up the Lingua language detector
detector = LanguageDetectorBuilder.from_all_languages().build()

# Initialize counters for tracking removal reasons
removal_reasons = {
    "too_short": 0,
    "non_english": 0,
    "special_characters": 0
}

def detect_language(text):
    language = detector.detect_language_of(text)
    if language:
        return str(language.iso_code_639_1).lower()  # Convert Enum to string and make it lowercase
    return "unknown"


def is_noisy_review(review_text):

    global removal_reasons

    # Check if review is empty or too short
    if not review_text or len(review_text.strip()) <= 3:
        removal_reasons["too_short"] += 1
        return True

    # Detect language using Lingua; mark as noisy if it's not English
    try:
        language = detect_language(review_text)
        if language != "isocode639_1.en":
            #print(language + ": " + review_text)  # Log non-English reviews
            removal_reasons["non_english"] += 1
            return True
    except Exception as e:
        # Handle potential issues with language detection
        print(f"Error detecting language for text: {review_text[:30]}... {str(e)}")
        removal_reasons["non_english"] += 1
        return True

    # Check for excessive special characters (could indicate strange encoding)
    if len(re.findall(r'[^\w\s]', review_text)) / len(review_text) > 0.5:
        removal_reasons["special_characters"] += 1
        return True

    return False

def normalize_review_text(text):
    if not text:
        return text
    # Normalize to NFC (Normalization Form C)
    normalized_text = unicodedata.normalize('NFC', text)
    # Remove non-printable characters
    cleaned_text = ''.join(c for c in normalized_text if c.isprintable())
    return cleaned_text

def clean_reviews(reviews):
    cleaned_reviews = []
    for review in reviews:
        if "review" in review:
            # Normalize text
            review["review"] = normalize_review_text(review["review"])
            # Check if the review is noisy
            if not is_noisy_review(review["review"]):
                cleaned_reviews.append(review)
    return cleaned_reviews

def process_json_files(input_folder, output_folder):
    global removal_reasons

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for file_name in os.listdir(input_folder):
        if file_name.endswith(".json"):
            input_file_path = os.path.join(input_folder, file_name)
            output_file_path = os.path.join(output_folder, file_name)

            # Load JSON file
            with open(input_file_path, 'r', encoding='utf-8') as infile:
                data = json.load(infile)

            # Initialize counts
            total_reviews_before = 0
            total_reviews_after = 0

            # Reset counters for removal reasons
            for key in removal_reasons.keys():
                removal_reasons[key] = 0

            # Process 'reviews' subarray
            for item in data:
                if "reviews" in item:
                    total_reviews_before += len(item["reviews"])
                    item["reviews"] = clean_reviews(item["reviews"])
                    total_reviews_after += len(item["reviews"])

            # Save cleaned data to output folder
            with open(output_file_path, 'w', encoding='utf-8') as outfile:
                json.dump(data, outfile, ensure_ascii=False, indent=4)

            # Print traces
            print(f"Processed file: {file_name}")
            print(f"Total reviews before cleaning: {total_reviews_before}")
            print(f"Total reviews after cleaning: {total_reviews_after}")
            print("Reviews removed due to:")
            for reason, count in removal_reasons.items():
                print(f"  - {reason.replace('_', ' ').capitalize()}: {count}")
            print("-" * 50)

def main():
    parser = argparse.ArgumentParser(description="Clean noisy reviews and normalize encoding in JSON files.")
    parser.add_argument("input_folder", type=str, help="Path to the input folder containing JSON files.")
    parser.add_argument("output_folder", type=str, help="Path to the output folder for cleaned JSON files.")
    args = parser.parse_args()

    process_json_files(args.input_folder, args.output_folder)

if __name__ == "__main__":
    main()
