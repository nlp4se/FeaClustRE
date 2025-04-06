# FeaClustRE: Feature Clustering and Analysis Visualization Tool

## Overview

**FeaClustRE** (Feature Clustering and Analysis Visualization Tool) is an advanced microservice that performs hierarchical clustering (HCI) and visualization of structured feature data using modern NLP and LLM techniques. It's designed to help you analyze and explore complex feature sets extracted from user reviews or other domain-specific texts.

This tool is part of the **RE-Miner Ecosystem**, which can be explored in the [GESSI-NLP4SE repository](https://github.com/gessi-nlp4se).

### Key Features

-  **Custom Clustering Algorithm** – Hand-made affinity-based clustering for grouping similar features.
-  **Dendrogram Visualization** – Hierarchical cluster visualizations for exploring feature relationships.
- **Preprocessing Pipelines** – Feature extraction, transformation, and normalization.
- **API & CLI Interface** – Supports both REST API calls and CLI-based workflows.
- **Hugging Face Integration** – Uses Meta’s LLaMA for embedding-based clustering (token required).
- **Docker-Ready** – Easily deployable via Docker for local or server environments.

---

## Table of Contents

1. [Installation](#installation)
2. [Configuration](#configuration)
3. [🔑 Hugging Face Token Authentication & LLaMA Access](#hugging-face-token-authentication--llama-access)
4. [Data Structure](#data-structure)
5. [API Usage](#api-usage)
6. [Request Parameters](#request-parameters)
7. [Response Format](#response-format)
8. [Examples](#examples)
9. [Flask Local Run](#flask-local-run)
10. [Docker Deployment](#docker-deployment)
11. [Troubleshooting](#troubleshooting)

---

## Installation

### Prerequisites

- Python 3.9+
- pipenv
- Docker (optional for container deployment)

### Steps

```bash
# Clone the repo
git clone https://github.com/your-org/feature-clustering-service.git
cd feature-clustering-service

# Install dependencies
pip install pipenv
pipenv install --deploy
pipenv run pip install torch --index-url https://download.pytorch.org/whl/cpu
pipenv run python -m spacy download en_core_web_sm
```

---

## Configuration

### Required `.env` File

Create a `.env` file in the root directory with the following contents:

```env
DG_SERVICE_URL=http://localhost
DG_SERVICE_PORT=3008
HUGGING_FACE_HUB_TOKEN=<Token>
```

---

## 🔑 Hugging Face Token Authentication & LLaMA Access

This project uses **Meta's LLaMA model**, which is gated and requires manual approval from Hugging Face.

### How to Get Access to LLaMA

1. Go to the [LLaMA Model 3.2-3B](https://huggingface.co/meta-llama) page.
2. Click **Request Access** and complete the form.
3. Wait for Hugging Face to approve access.

### Using Your Token

Once approved:

1. Add your Hugging Face token in the `.env` file as shown above.
2. The backend will use this token to authenticate with Hugging Face's API.

---

## Data Structure

### Directory Layout

```
data/
├── Stage 1 - Data Collection/
│   └── raw_data/                    # Raw CSV data
│
├── Stage 2 - Hierarchical Clustering/
│   ├── input/                       # Input features for clustering
│   ├── output/                      # .pkl files with dendrograms
│   └── preprocessed_features_jsons/ # JSON versions of features
│
└── Stage 3 - Topic Modelling/
    ├── input/                       # Stage 2 output as input
    └── output/                      # Final results and visualizations
        ├── cluster_summaries/
        ├── dendrograms/
        └── hierarchies/
```

### File Types Table

| Stage | Directory | File Type | Description |
|-------|-----------|-----------|-------------|
| 1 | raw_data/ | `.csv` | Raw input feature data |
| 2 | preprocessed_features_jsons/ | `.json` | Preprocessed feature representations |
| 2 | output/ | `.pkl` | Pickled dendrogram clustering models |
| 3 | output/dendrograms/ | `.png` | Dendrogram visualizations |
| 3 | output/hierarchies/ | `.json` | Final cluster trees |
| 3 | output/cluster_summaries/ | `.csv` | Summary stats per cluster |

---

## Example Input CSV

Sample format for raw CSV:

```csv
app_name,package_name,category,review_id,review_text
"Discord - Talk, Play, Hang Out",com.discord,COMMUNICATION,6b6e58c3-81c3-4fce-9b0d-b619be49f156,"This is very very usefull app please try it"
"Discord - Talk, Play, Hang Out",com.discord,COMMUNICATION,00280421-44e5-4026-8374-72b714bfe6ec,"Buggy (eg. notifications just don't work for me)..."
"Discord - Talk, Play, Hang Out",com.discord,COMMUNICATION,b4f03728-9288-4c8c-a928-9b17ce651105,"it's ok. discord is a narc, but..."
...
```

Ensure it contains a `review_text` column with meaningful content.

---

## API Usage

### Endpoint

```
POST /generate_kg
```

### Request Format

- `multipart/form-data`
- Include your CSV under the `file` field.

---

## Request Parameters

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `preprocessing` | boolean | `false` | Enable feature preprocessing |
| `affinity` | string | `bert` | Options: `bert`, `paraphrase`, `tf-idf` |
| `metric` | string | `cosine` | Distance metric |
| `threshold` | float | `0.2` | Clustering threshold |
| `linkage` | string | `average` | Clustering method |
| `obj-weight` | float | `0.25` | Weight of object embeddings |
| `verb-weight` | float | `0.75` | Weight of verb embeddings |
| `app_name` | string | `''` | Name of the application |

---

## Response Format

```json
{
  "message": "Dendrogram generated successfully",
  "dendrogram_path": "path/to/generated/file.pkl"
}
```

---

## Examples

### cURL

```bash
curl -X POST \
  "http://localhost:3008/generate_kg?preprocessing=true&affinity=bert&threshold=0.2&linkage=average&obj-weight=0.25&verb-weight=0.75&app_name=Bard" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@features.csv"
```

### Python

```python
import requests

params = {
    "preprocessing": "true",
    "affinity": "bert",
    "threshold": 0.2,
    "linkage": "average",
    "obj-weight": 0.25,
    "verb-weight": 0.75,
    "app_name": "Bard"
}
files = {"file": open("features.csv", "rb")}
res = requests.post("http://localhost:3008/generate_kg", params=params, files=files)
print(res.json())
```

---

## Flask Local Run

To run locally via Flask:

```bash
pipenv run python app.py
```

You should see:

```
Running on http://127.0.0.1:3008
```

---

## Docker Deployment

### Build the Docker Image

```bash
docker build -t feaclustre-service .
```

### Run the Container

```bash
docker run -p 3008:3008 --env-file .env feaclustre-service
```

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `TokenError` from Hugging Face | Make sure your token is in `.env` and you have access to LLaMA |
| Invalid CSV | Ensure `review_text` column is present and clean |
| Memory Errors | Try smaller batch sizes or fewer features |
| Docker Port Already Used | Change `DG_SERVICE_PORT` or bind to another local port |

---
