import os
import requests
import glob

# Configurable parameters
INPUT_FOLDER = r'C:\Users\Max\NLP4RE\Dendogram-Generator\data\Stage 2 - Hierarchical Clustering\input'
BASE_URL = 'http://127.0.0.1:3008/dendogram/generate_kg'

DEFAULT_PARAMS = {
    'preprocessing': 'true',
    'affinity': 'bert',
    'metric': 'cosine',
    'threshold': '0.2',
    'linkage': 'average',
    'obj-weight': '0.25',
    'verb-weight': '0.75'
}


def process_csv_files(input_folder, base_url, default_params):
    # Find all .csv files in the specified folder
    csv_files = glob.glob(os.path.join(input_folder, '*.csv'))

    for csv_file in csv_files:
        try:
            # Extract app_name from the filename (<app_name>.csv)
            app_name = os.path.splitext(os.path.basename(csv_file))[0]
            params = default_params.copy()
            params['app_name'] = app_name

            # Prepare the file to be uploaded
            with open(csv_file, 'rb') as f:
                files = {'file': (os.path.basename(csv_file), f, 'text/csv')}

                # Send a POST request to the API
                response = requests.post(base_url, params=params, files=files)

                if response.status_code == 200:
                    print(f"Successfully processed {csv_file}")
                else:
                    print(f"Failed to process {csv_file}: {response.status_code} - {response.text}")

        except Exception as e:
            print(f"Error processing {csv_file}: {e}")


if __name__ == '__main__':
    # Run the script
    process_csv_files(INPUT_FOLDER, BASE_URL, DEFAULT_PARAMS)
