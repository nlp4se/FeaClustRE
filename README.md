# FeaClustRE – A Feature Clustering and Analysis Visualization Tool

---

## Introduction
FeaClustRE (**Feature Clustering and Analysis Visualization Tool**) is an advanced tool designed to **analyze, cluster, and visualize structured data features** using machine learning techniques. It provides **hierarchical clustering, dendrogram visualizations, and feature-based insights** to help researchers and data scientists explore complex datasets.

This tool uses **Meta's LLaMA model** for feature embedding and **Hugging Face's Transformers** for feature family clustering. 

With a flexible **backend API**, a **CLI client**, and **visualization tools**, FeaClustRE supports both **interactive analysis and automated batch processing**.

### Key Features
- **Custom Clustering Algorithm** – Uses a hand-made affinity-based clustering approach to automatically group similar features.
- **Dendrogram Visualization** – Generates hierarchical visualizations to explore feature relationships.
- **Preprocessing Pipelines** – Provides data cleaning and transformation utilities.
- **API and CLI Support** – Run analysis through API endpoints or via local CLI commands.
- **Hugging Face Model Integration** – Supports **Meta LLaMA** for embedding-based clustering (requires access).
- **Docker Support** – Easily deployable using **Docker and Docker Compose**.
---

## 📌 Table of Contents
- [Demo & Screenshots](#demo--screenshots)
- [Hugging Face Token Authentication & LLaMA Access](#hugging-face-token-authentication--llama-access)
- [Installation](#installation)
  - [Local Installation](#local-installation)
  - [Docker Installation](#docker-installation)
- [Project Structure](#project-structure)
- [Running Preprocessing Scripts](#running-preprocessing-scripts)

---

## 🎥 Demo & Screenshots
_(Coming Soon)_

---



## 🔑 Hugging Face Token Authentication & LLaMA Access

This project uses **Meta's LLaMA model**, which is **gated** and requires **manual approval** from Hugging Face.

### **How to Get Access to LLaMA**
1. Visit the [LLaMA Model 3.2-3B Page](https://huggingface.co/meta-llama/Llama-3.2-3B).
2. Click **Request Access** and follow the instructions.
3. Wait for Hugging Face to approve your request.

### **Using Your Hugging Face Token**
To authenticate, you **must set your Hugging Face token** before running the project.

#### **1️⃣ Set the Token in `.env`**
In the `.env` file in the project root, add:

```
HUGGING_FACE_HUB_TOKEN=your_huggingface_token
```

---

## 🛠 Installation

### Local Installation
1) **Before using, install the required spaCy model**:
```sh
python -m spacy download en_core_web_sm
```

2) **Set your `HUGGING_FACE_HUB_TOKEN` in the .env file**
```
HUGGING_FACE_HUB_TOKEN=${HUGGINGFACE_TOKEN}
```
3) **Install dependencies**
```sh
pipenv install
```
4) **Execute API**
```sh
flask run --port=3008
```

### Docker Installation
1) **Build and run the Docker Image**
```sh
docker build -t release . && docker run -p 3008:3008 --name feaclustre release 
```

---

## 📂 Project Structure
The following is the structure of the FeaClustRE project:

```
FeaClustRE/
│── .github/                  # GitHub Actions & CI/CD workflows
│── backend/                   # Backend services and clustering algorithms
│   │── data-preprocessing/     # Scripts for processing raw data
│   │── Affinity_strategy.py    # Strategy for affinity clustering
│   │── Context.py              # Context manager for clustering
│   │── dendogram_controller.py # Handles dendrogram API calls
│   │── dendogram_service.py    # Service for generating dendrograms
│   │── graph_controller.py     # Graph visualization API
│   │── graph_service.py        # Graph computation logic
│   │── preprocessing_service.py # Handles feature preprocessing
│   │── tf_idf_utils.py         # Utilities for TF-IDF calculations
│   │── utils.py                # General utility functions
│   │── visualization_service.py # Generates visualizations for clusters
│── cli-client/                 # Command-line interface for clustering
│   │── scripts/                # Helper scripts
│   │── dendogram_generation.py # CLI tool for dendrogram generation
│   │── dynamic_visualizator.py # CLI tool for dynamic visualization
│   │── requester.py            # Request handler for API calls
│   │── visualizator.py         # CLI tool for visualization
│── data/                       # Data storage directory
│── .env                        # Environment variables (ignored in Git)
│── .gitattributes              # Git attributes
│── .gitignore                  # Git ignore file
│── docker-compose.yml          # Docker Compose configuration
│── Dockerfile                  # Docker build configuration
│── Pipfile                     # Pipenv dependencies
│── Pipfile.lock                 # Locked dependencies
│── README.md                    # Project documentation
│── wsgi.py                      # Entry point for the Flask application
```

---
