# Use Python 3.9 as base image
FROM python:3.9

WORKDIR /wsgi

# Copy only dependency files first to leverage Docker caching
COPY Pipfile Pipfile.lock /wsgi/

# Install Pipenv and dependencies
RUN pip install pipenv && pipenv install --deploy --ignore-pipfile

# Install CPU-only PyTorch (avoids CUDA errors)
RUN pipenv run pip install torch --index-url https://download.pytorch.org/whl/cpu

# Download spaCy model inside the virtual environment
RUN pipenv run python -m spacy download en_core_web_sm

# Copy the application files
COPY . /wsgi

# Unset HF_TOKEN to prevent conflicts
ENV HF_TOKEN=""

# Expose the port for Flask
EXPOSE 3008

# Run Flask
CMD ["pipenv", "run", "flask", "run", "--host=0.0.0.0", "--port=3008"]
