# FeaClustRE – A Feature Clustering and Analysis Visualization Tool
---

Documentation writing in progress...

---

## Demo & Screenshots

---
## Installation



### Hugging Face Token Authentication & LLaMA Access

This project uses **Meta's LLaMA model**, which is **gated** and requires **manual approval** from Hugging Face.

#### **How to Get Access to LLaMA**
1. Visit the [LLaMA Model 3.2-3B Page](https://huggingface.co/meta-llama/Llama-3.2-3B).
2. Click **Request Access** and follow the instructions.
3. Wait for Hugging Face to approve your request.

#### **Using Your Hugging Face Token**
To authenticate, you **must set your Hugging Face token** before running the project.

In the `.env` file in the project root add:

```
HUGGING_FACE_HUB_TOKEN=your_huggingface_token
```
---

### Local Installation

Before using, install the required spaCy model:
```sh
python -m spacy download en_core_web_sm
```
### Docker Installation

---



