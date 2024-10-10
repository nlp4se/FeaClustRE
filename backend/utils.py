import os
import pandas as pd

MODEL_DIRECTORY_PATH = 'static' + os.path.sep + 'pkls'
MODEL_DIRECTORY_CSV_PATH = 'static' + os.path.sep + 'csv'
MODEL_DIRECTORY_CSV_EMBEDDIGNS_PATH = 'static' + os.path.sep + 'csv' + os.path.sep + 'embeddings'


class Utils:

    @staticmethod
    def save_to_csv(dataframe: pd.DataFrame, csv_filename: str):
        if not os.path.exists(MODEL_DIRECTORY_CSV_PATH):
            os.makedirs(MODEL_DIRECTORY_CSV_PATH)
        csv_file_path = os.path.join(os.getcwd(), MODEL_DIRECTORY_CSV_PATH, csv_filename)
        print(f"Saving CSV to {csv_file_path}...")
        dataframe.to_csv(csv_file_path, index=False)
        return csv_file_path

    @staticmethod
    def save_to_pkl(model_info: dict, pkl_filename: str):
        if not os.path.exists(MODEL_DIRECTORY_PATH):
            os.makedirs(MODEL_DIRECTORY_PATH)
        pkl_file_path = os.path.join(os.getcwd(), MODEL_DIRECTORY_PATH, pkl_filename)
        print(f"Saving model to {pkl_file_path}...")
        joblib.dump(model_info, pkl_file_path)
        return pkl_file_path

    @staticmethod
    def generate_pkl(application_name, clustering_model, model_name, dense_data_array, labels, distance_threshold, linkage, metric, verb_weight, object_weight):
        print("Saving clustering metadata for plotting...")
        model_info = {
            'affinity': f'{model_name} {metric} {linkage}',
            'labels': labels,
            'model_name': model_name,
            'model': clustering_model,
            'data_points': dense_data_array,
            'application_name': application_name,
            'distance_threshold': distance_threshold,
            'verb_weight': verb_weight,
            'object_weight': object_weight
        }

        if hasattr(clustering_model, 'cluster_centers_'):
            model_info['cluster_centers'] = clustering_model.cluster_centers_

        pkl_file_name = f"{application_name}_{model_name.lower()}_{metric}_{linkage}_thr-{distance_threshold}_vw-{verb_weight}_ow-{object_weight}.pkl"
        return Utils.save_to_pkl(model_info, pkl_file_name)
