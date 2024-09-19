import os
import subprocess

folder_path = r'C:\Users\Max\NLP4RE\Dendogram-Generator\static\pkls'

# Define the path to the visualizator.py script
visualizator_script = r'visualizator.py'

# Iterate over each file in the folder
for filename in os.listdir(folder_path):
    # Construct the full path to the file
    file_path = os.path.join(folder_path, filename)

    # Ensure it's a file and not a directory
    if os.path.isfile(file_path):
        # Execute the visualizator.py script with the file as an argument
        subprocess.run(['python', visualizator_script, file_path], check=True)
