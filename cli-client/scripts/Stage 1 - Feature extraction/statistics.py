import argparse
import pandas as pd
import os
import re
from collections import Counter
import spacy
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

def parse_pascal_case(text):
    """Convert PascalCase to space-separated lowercase words"""
    return re.sub(r'(?<!^)(?=[A-Z])', ' ', text).lower()

def process_files(input_folder, output_folder, top_n):
    # Load spaCy English model
    nlp = spacy.load('en_core_web_sm')
    
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Dictionary to store features by category
    category_features = {}
    
    # Process each CSV file in the input folder
    for filename in os.listdir(input_folder):
        if not filename.endswith('.model_embeddings'):
            continue
            
        input_path = os.path.join(input_folder, filename)
        base_name = os.path.splitext(filename)[0]
        
        # Read CSV file
        df = pd.read_csv(input_path)
        
        # Extract and flatten features
        all_features = []
        for cell in df['extracted_features'].dropna():
            features = cell.split(';')
            all_features.extend(features)
            
        # Parse features from PascalCase
        parsed_features = [parse_pascal_case(f) for f in all_features]
        
        # Create statistics
        feature_counts = Counter(parsed_features)
        stats_df = pd.DataFrame({
            'feature': list(feature_counts.keys()),
            'count': list(feature_counts.values())
        }).sort_values('count', ascending=False)
        
        # Save individual file statistics
        stats_output = os.path.join(output_folder, f"{base_name}_statistics.model_embeddings")
        stats_df.to_csv(stats_output, index=False)
        
        # Process with spaCy for POS tagging
        pos_data = []
        for feature in parsed_features:
            doc = nlp(feature)
            pos_data.extend([(token.text, token.pos_) for token in doc])
            
        # Extract verbs and nouns
        verb_tags = {'VERB', 'AUX'}
        noun_tags = {'NOUN', 'PROPN'}
        
        verbs = [word for word, pos in pos_data if pos in verb_tags]
        nouns = [word for word, pos in pos_data if pos in noun_tags]
        
        # Get top N most common verbs and nouns
        top_verbs = Counter(verbs).most_common(top_n)
        top_nouns = Counter(nouns).most_common(top_n)
        
        # Create cross-tabulation matrix
        matrix_data = []
        for feature in parsed_features:
            doc = nlp(feature)
            feature_verbs = [token.text for token in doc if token.pos_ in verb_tags]
            feature_nouns = [token.text for token in doc if token.pos_ in noun_tags]
            
            for verb in feature_verbs:
                for noun in feature_nouns:
                    matrix_data.append((verb, noun))
        
        # Create heatmap matrix
        verb_dict = dict(top_verbs)
        noun_dict = dict(top_nouns)
        
        heatmap_data = pd.DataFrame(0, 
                                  index=verb_dict.keys(),
                                  columns=noun_dict.keys())
        
        for verb, noun in matrix_data:
            if verb in verb_dict and noun in noun_dict:
                heatmap_data.loc[verb, noun] += 1
        
        # Apply log1p transformation (log(1+x)) to the data
        heatmap_data_log = np.log1p(heatmap_data)
        
        # Create and save heatmap with transformed data
        plt.figure(figsize=(15, 10))
        sns.heatmap(heatmap_data_log, cmap='YlOrRd')
        plt.title(f'Verb-Noun Heatmap for {base_name} (log scale)')
        plt.tight_layout()
        
        heatmap_output = os.path.join(output_folder, f"{base_name}_heatmap.png")
        plt.savefig(heatmap_output)
        plt.close()
        
        # Group features by category
        for category, group in df.groupby('app_categoryId'):
            if category not in category_features:
                category_features[category] = []
                
            # Extract and flatten features for this category
            for cell in group['extracted_features'].dropna():
                features = cell.split(';')
                category_features[category].extend(features)
    
    # Process each category's features
    for category, features in category_features.items():
        # Parse features from PascalCase
        parsed_features = [parse_pascal_case(f) for f in features]
        
        # Create statistics
        feature_counts = Counter(parsed_features)
        stats_df = pd.DataFrame({
            'feature': list(feature_counts.keys()),
            'count': list(feature_counts.values())
        }).sort_values('count', ascending=False)
        
        # Save category statistics
        stats_output = os.path.join(output_folder, f"category_{category}_statistics.model_embeddings")
        stats_df.to_csv(stats_output, index=False)
        
        # Process with spaCy for POS tagging
        pos_data = []
        for feature in parsed_features:
            doc = nlp(feature)
            pos_data.extend([(token.text, token.pos_) for token in doc])
            
        # Extract verbs and nouns
        verb_tags = {'VERB', 'AUX'}
        noun_tags = {'NOUN', 'PROPN'}
        
        verbs = [word for word, pos in pos_data if pos in verb_tags]
        nouns = [word for word, pos in pos_data if pos in noun_tags]
        
        # Get top N most common verbs and nouns
        top_verbs = Counter(verbs).most_common(top_n)
        top_nouns = Counter(nouns).most_common(top_n)
        
        # Create cross-tabulation matrix
        matrix_data = []
        for feature in parsed_features:
            doc = nlp(feature)
            feature_verbs = [token.text for token in doc if token.pos_ in verb_tags]
            feature_nouns = [token.text for token in doc if token.pos_ in noun_tags]
            
            for verb in feature_verbs:
                for noun in feature_nouns:
                    matrix_data.append((verb, noun))
        
        # Create heatmap matrix
        verb_dict = dict(top_verbs)
        noun_dict = dict(top_nouns)
        
        heatmap_data = pd.DataFrame(0, 
                                  index=verb_dict.keys(),
                                  columns=noun_dict.keys())
        
        for verb, noun in matrix_data:
            if verb in verb_dict and noun in noun_dict:
                heatmap_data.loc[verb, noun] += 1
        
        # Apply log1p transformation (log(1+x)) to the data
        heatmap_data_log = np.log1p(heatmap_data)
        
        # Create and save heatmap with transformed data
        plt.figure(figsize=(15, 10))
        sns.heatmap(heatmap_data_log, cmap='YlOrRd')
        plt.title(f'Verb-Noun Heatmap for Category {category} (log scale)')
        plt.tight_layout()
        
        heatmap_output = os.path.join(output_folder, f"category_{category}_heatmap.png")
        plt.savefig(heatmap_output)
        plt.close()

def main():
    parser = argparse.ArgumentParser(description='Process feature extraction files')
    parser.add_argument('--input_folder', required=True, help='Input folder containing CSV files')
    parser.add_argument('--output_folder', required=True, help='Output folder for statistics and heatmaps')
    parser.add_argument('--top_n', type=int, default=50, help='Number of top verbs and nouns to extract')
    
    args = parser.parse_args()
    process_files(args.input_folder, args.output_folder, args.top_n)

if __name__ == "__main__":
    main()
