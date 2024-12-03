import os
import csv
import spacy
from transformers import pipeline
import argparse
import pandas as pd


def extract_features_from_review(review, ner_pipeline):
    """
    Extract features from the review using the NER model.
    """
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(review)
    sentences = [sent.text for sent in doc.sents]

    extracted_features = []
    for sentence in sentences:
        ner_results = ner_pipeline(sentence)
        current_feature = []
        for entity in ner_results:
            if (len(entity['word']) > 2):
                extracted_features.append(entity['word'])
            # print(entity)
            # if entity['entity'] == 'B-feature':
                # Start of a new feature
                # if current_feature:
                    # extracted_features.append(" ".join(current_feature))
                # current_feature = [entity['word']]
            # elif entity['entity'] == 'I-feature':
                # Continuation of a feature
                # current_feature.append(entity['word'])
        # Append the last feature, if any
        # if current_feature:
            # extracted_features.append(" ".join(current_feature))

    return ";".join(extracted_features)


def process_csv(input_file, output_file, ner_pipeline):
    """
    Process the input CSV file to extract features and save the updated file.
    """
    # Read the input CSV
    df = pd.read_csv(input_file)

    # Extract features for each review
    features = []
    for index, row in df.iterrows():
        review = row.get('review', '')
        if pd.isna(review) or not isinstance(review, str):
            features.append('')
        else:
            features.append(extract_features_from_review(review, ner_pipeline))

    # Add the extracted features as a new column
    df['extracted_features'] = features

    # Save the updated DataFrame to the output file
    df.to_csv(output_file, index=False)
    print(f"Processed file saved to {output_file}")


def process_folder(input_folder, output_folder):
    """
    Process all CSV files in the input folder.
    """
    # Load the NER pipeline
    ner_pipeline = pipeline("ner", model="quim-motger/t-frex-xlnet-large-cased", aggregation_strategy="simple")

    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Process each CSV file in the input folder
    for file_name in os.listdir(input_folder):
        if file_name.endswith('.csv'):
            input_file = os.path.join(input_folder, file_name)
            output_file = os.path.join(output_folder, file_name)
            print(f"Processing {input_file} -> {output_file}")
            process_csv(input_file, output_file, ner_pipeline)

    print(f"All files processed. Results saved in {output_folder}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract features from reviews using NER model")
    parser.add_argument('--input_folder', required=True, help='Path to the input folder containing CSV files')
    parser.add_argument('--output_folder', required=True, help='Path to the output folder to save updated CSV files')

    args = parser.parse_args()
    process_folder(args.input_folder, args.output_folder)
