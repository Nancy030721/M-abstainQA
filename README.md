# Reproducing "Teaching LLMs to Abstain across Languages via Multilingual Feedback"

This repository contains our implementation for reproducing the experiments from the paper:
["Teaching LLMs to Abstain across Languages via Multilingual Feedback"](https://arxiv.org/abs/2406.15948).

## 📌 Overview
This project explores multilingual feedback as a strategy for improving the abstention behavior of large language models (LLMs), especially in low-resource languages. We implement the **MONO-NATIVE**, **MONO-NATIVE**, **MULTI-RANDOM**, and **MULTI-RELATED** feedback approach and compare it with other abstention strategies.

## 🔧 Implementation Details
- **Language:** Python
- **Key Libraries:**
  - [Hugging Face Transformers](https://huggingface.co/docs/transformers/index) – LLM interaction
  - [PyTorch](https://pytorch.org/) – Model inference and computations
  - [OpenAI API](https://platform.openai.com/docs/) - Accessing GPT-4
  - [Google Generative AI](https://ai.google.dev/) - Accessing Gemini models
- **Datasets Used:**
  - [M-MMLU](https://github.com/nlp-uoregon/mlmm-evaluation)(Multilingual Massive Multitask Language Understanding): Open-sourced and accessible for multilingual QA evaluation
  - [HellaSwag](https://github.com/nlp-uoregon/mlmm-evaluation): Open-sourced dataset for commonsense reasoning in multiple languages.
  <!-- - [Belebele](https://huggingface.co/datasets/belebele): Multilingual reading comprehension dataset -->
## 📊 Dataset Statistics
Below are the specific languages and datasets we used in our experiments. Other languages available in these datasets were not included in our analysis:

| Language |  code | HellaSwag Test Data | MMLU Test Data |
|----------|---------|---------------------|----------------|
| Bengali  |   bn  |       771    |     726 |
| Kannada  |   kn  |       761    |     646 |
| Malayalam|   ml  |       753    |     639 |
| Marathi  |   mr  |       774    |     720 |
| Nepali |   ne  |       777    |     740 |
| Tamil  |   ta  |       700    |     676 |
| Telugu |   te  |       738    |     647 |

## How to Run 🚀 
1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo-name.git
   cd your-repo-name
2. Install dependencies:
   ```bash
   git pip install -r requirements.txt
To run the **MultiRandom** experiment, use the following command:
```bash
python multirandom.py -m <model_name> -d <dataset> -s <language> -o <portion> -l -f
```
#### Example:
```bash
python multirandom.py -m gemini -d mmlu -s bn -o 0.01 -l -f
```
#### Explanation:
- `-m gemini` → Specifies the language model to use (e.g., `gemini`).
- `-d mmlu` → Specifies the dataset to use (e.g., `mmlu`).
- `-s bn` → Sets the primary language code (`bn` for Bengali).
- `-o 0.01` → Uses only 1% of the dataset for quick testing.
- `-l` → Saves a local JSON file of predictions.
- `-f` → Saves a separate file with generated feedback.

### Running the MultiRelated Experiment
The **MultiRelated** experiment follows the same structure as MultiRandom. To run it, use:
```bash
python multirelated.py -m <model_name> -d <dataset> -s <language> -o <portion> -l -f
```
#### Example:
```bash
python multirelated.py -m gemini -d mmlu -s bn -o 0.01 -l -f
```
This command follows the same argument structure as `multirandom.py`, with the difference being that it runs the MultiRelated experiment instead.

Both experiments allow flexibility in model selection, dataset choice, and language specification. The `-o` parameter enables running on a subset of data for efficiency during testing.

We are still in the process of implementing the other two methods. Once completed, instructions will be updated here.

## Test Local LLM ([Aya-101](https://huggingface.co/CohereForAI/aya-101))

To reproduce the experiments with Aya-13B in the original paper, we implemented offline batched inference given our constrained computing resources and time. We used [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM) as the serving engine, and largely reduced the experiment time by ~10x compared to the original implementation with native [Transformers](https://github.com/huggingface/transformers) inference.

First we need to prepare the model and the engine. Detailed instructions can be found [here](https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/enc_dec/README.md). With [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM) and necessary dependencies installed in your environment, we need the model converting script as provided by official [examples](https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/enc_dec/convert_checkpoint.py): 
```bash
# Download the model, your local directory may vary
huggingface-cli download CohereForAI/aya-101 --local-dir /data/aya-101

# Convert the model to TensorRT checkpoint
export INFERENCE_PRECISION="bfloat16" # We quantize to bf16 to save some memory
python convert_checkpoint.py --model_type "t5" \
                --model_dir /data/aya-101 \
                --output_dir /data/aya-101-trt \
                --dtype ${INFERENCE_PRECISION} \
                --context_fmha disable # Fused Multi-Head Attention not supported for T5

# Build the TensorRT engine
trtllm-build --checkpoint_dir /data/aya-101-trt-bf16/encoder \
                --output_dir  /data/aya-101-trt-bf16-engine/encoder \
                --paged_kv_cache disable \
                --moe_plugin disable \
                --max_beam_width 1 \
                --max_batch_size 4 \
                --max_input_len 4864 \ 
                --max_num_tokens 5120 \ # Sizes fits for batch_size=4, max_new_tokens=200, single H100 serving
                --gemm_plugin ${INFERENCE_PRECISION} \
                --bert_attention_plugin ${INFERENCE_PRECISION} \
                --gpt_attention_plugin ${INFERENCE_PRECISION} \
                --remove_input_padding enable \
                --context_fmha disable

trtllm-build --checkpoint_dir /data/aya-101-trt-bf16/decoder \
                --output_dir /data/aya-101-trt-bf16-engine/decoder \ 
                --moe_plugin disable \
                --max_beam_width 1 \
                --max_batch_size 4 \
                --max_input_len 1 \
                --max_seq_len 201 \
                --max_encoder_input_len 4864 \  # Match with encoder
                --gemm_plugin ${INFERENCE_PRECISION} \
                --bert_attention_plugin ${INFERENCE_PRECISION} \
                --gpt_attention_plugin ${INFERENCE_PRECISION} \
                --remove_input_padding enable \
                --context_fmha disable
```
Make sure that your engine directory is properly set in [lm_utils.py](https://github.com/Nancy030721/M-abstainQA/blob/ecff080e96876b7e0d68682e1dcdc9f2966f7dec/lm_utils.py#L31), then you're good to go to run the experiments:
```bash
python multirelated_batched.py -m aya_13b -d mmlu -s bn -o 0.1 -l -f
```

## Results: MMLU Test Data Abstain Rate
Below is a comparison of our reproduced abstain rates with the original results from the paper.

| Language  | Code |MonoEnglish (Ours)|MonoNative (Ours)|MultiRandom (Ours)|MultiRelated (Ours)|MonoEnglish (Original)|MonoNative (Original) |MultiRandom (Original)|MultiRelated (Original)|Difference (MultiRelated)|
|-|-|-|-|-|-|-|-|-|-|-|
| Bengali   | bn | 0.377 | 0.375 | 0.552 | 0.661 | 0.580 | 0.611 | 0.597 | 0.621  | **+0.040**  |
| Kannada   | kn | 0.370 | 0.358 | 0.475 | 0.646 | 0.515 | 0.607 | 0.615 | 0.704  | **-0.058**  |
| Malayalam | ml | 0.316 | 0.356 | 0.388 | 0.642 | 0.604 | 0.649 | 0.561 | 0.595  | **+0.047**  |
| Marathi   | mr | 0.363 | 0.356 | 0.536 | 0.461 | 0.529 | 0.460 | 0.524 | 0.661  | **-0.200**  |
| Nepali    | ne | 0.343 | 0.359 | 0.462 | 0.345 | 0.578 | 0.583 | 0.549 | 0.590  | **-0.245**  |
| Tamil     | ta | 0.311 | 0.309 | 0.570 | 0.732 | 0.533 | 0.594 | 0.628 | 0.643  | **+0.089**  |
| Telugu    | te | 0.347 | 0.332 | 0.433 | 0.491 | 0.520 | 0.688 | 0.605 | 0.628  | **-0.137**  |
| **Average** | -  | **0.347** | **0.347** | **0.484** | **0.568** | **0.551** | **0.599** | **0.583** | **0.635** | **-0.067**
