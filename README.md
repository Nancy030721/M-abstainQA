# Reproducing "Teaching LLMs to Abstain across Languages via Multilingual Feedback"

This repository contains our implementation for reproducing the experiments from the paper:
["Teaching LLMs to Abstain across Languages via Multilingual Feedback"](https://arxiv.org/abs/2406.15948).

## 📌 Overview
This project explores multilingual feedback as a strategy for improving the abstention behavior of large language models (LLMs), especially in low-resource languages. We implement the **MULTI-RELATED** feedback approach and compare it with other abstention strategies.

## 🔧 Implementation Details
- **Language:** Python
- **Key Libraries:**
  - [Hugging Face Transformers](https://huggingface.co/docs/transformers/index) – LLM interaction
  - [PyTorch](https://pytorch.org/) – Model inference and computations
  - [OpenAI API](https://platform.openai.com/docs/) - Accessing GPT-4
- **Datasets Used:**
  - [M-MMLU](https://huggingface.co/datasets/m-mmlu)(Multilingual Massive Multitask Language Understanding): Open-sourced and accessible for multilingual QA evaluation
  - [HellaSwag](https://rowanzellers.com/hellaswag/): Open-sourced dataset for commonsense reasoning in multiple languages.
  - [Belebele](https://huggingface.co/datasets/belebele): Multilingual reading comprehension dataset
## 📊 Dataset Statistics
Below are the specific languages and datasets we used in our experiments. Other languages available in these datasets were not included in our analysis:

| Language | HellaSwag Test Data | MMLU Test Data |
|----------|--------------------|---------------|
| bn       | 771                | 726           |
| kn       | 761                | 646           |
| ml       | 753                | 639           |
| mr       | 774                | 720           |
| ne       | 777                | 740           |
| ta       | 700                | 676           |
| te       | 738                | 647           |

## 🚀 How to Run
1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo-name.git
   cd your-repo-name
2. Install dependencies:
   ```bash
   git pip install -r requirements.txt
3. Run experiments:
   We are still in the process of implementing the main experiment pipeline.
   Once completed, instructions will be updated here.
   
