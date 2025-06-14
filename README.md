# Scenario Mining with LLMs

This repository contains the code for our 3rd-place solution to the **Argoverse 2 Scenario Mining Challenge** at the **CVPR 2025 Workshop on Autonomous Driving**.  
Our method leverages Large Language Models (LLMs) to convert natural language scenario descriptions into executable Python scripts for filtering safety-critical scenarios from driving logs.

---

## 🛠 Environment Setup

Create and activate the Conda environment, then install the required dependencies:

```bash
conda create -n refav python=3.11
conda activate refav
pip install -r requirements.txt
pip install git+https://github.com/JonathonLuiten/TrackEval.git
```

---

## 📦 Data Preparation

Please follow the steps below to set up the required datasets.

### 1. Download Argoverse 2 Sensor Dataset

Download the official Argoverse 2 (AV2) sensor dataset from the [Argoverse website](https://www.argoverse.org/av2.html) and place it in the following directory:

```
data/av2/
```

### 2. Download Scenario Mining Dataset

Download the pre-processed [scenario mining dataset](https://huggingface.co/datasets/CainanD/RefAV/tree/main) (including descriptions and log ids).

Place the downloaded files in:

```
av2_sm_downloads/
```

### 3. Download Official Tracking Results

Download the [official tracking predictions](https://drive.google.com/file/d/1X19D5pBBO56eb_kvPOePLLhHDCsY0yql/view) used in the challenge:

After downloading, unzip and place the `.pkl` tracking results into:

```
tracker_downloads/
```

---

## 🚀 Getting Started

### 1. Offline Inference (Web-based LLMs)

Use web-based LLMs (e.g., [Gemini](https://aistudio.google.com/prompts/new_chat)) to generate code from descriptions.

#### ✅ Step 1: Aggregate Descriptions

```bash
python run/process_json_pro.py av2_sm_downloads/log_prompt_pairs_test.json -o refAV/llm_prompting/all_descriptions.txt
```

#### ✅ Step 2: Construct Instruction Prompt

Fill in the following template using:

- `{refav_context}` → from `refAV/llm_prompting/funcs.txt`
- `{av2_categories}` → from `refAV/llm_prompting/categories.txt`
- `{description_file}` → from the output of Step 1
- `{prediction_examples}` → from `refAV/llm_prompting/examples.txt`

```text
Please use the following functions to find instances of a referred object in an autonomous driving dataset. Be precise to the description, try to avoid returning false positives.

atomic functions:
{refav_context}

categories:
{av2_categories}

natural language descriptions:
{description_file}

Here is a list of examples:
{prediction_examples}

Output all the description and code pairs. Wrap all code in one python block and do not provide alternatives. Output code even if the given functions are not expressive enough to find the scenario.
```

#### ✅ Step 3: Generate and Save Code

Copy the instruction into your selected LLM. Save the returned code into:

```
./generated_code.txt
```

#### ✅ Step 4: Validate & Split

```bash
python run/check_and_split.py refAV/llm_prompting/all_descriptions.txt generated_code.txt output/llm_code_predictions/$LLM
```

#### ✅ Step 5: Run Inference

```bash
python run_experiment.py --exp_name $exp_name \
  --tracker_predictions_pkl tracker_downloads/Le3DE2E_tracking_predictions_val.pkl \
  --procs_per_task 3
```

#### ✅ Step 6: Collect Errors

```bash
python consolidate_errors.py \
  refav/output/sm_predictions/$exp_name/results/errors \
  refav/output/sm_predictions/$exp_name/results/errors_consolidate.txt
```

#### ✅ Step 7: Retry Failed Descriptions

Repeat Steps 2–6 with the failed description file until all scenarios are handled.

---

### 2. Online Inference (Open-source LLMs)

To use an open-source LLM (e.g., Qwen2.5) for full offline inference:

#### ✅ Step 1: Download the Model

```bash
huggingface-cli download Qwen/Qwen2.5-7B --local-dir checkpoints/Qwen2.5-7B
```

#### ✅ Step 2: Run Inference with Local Model

```bash
python run_experiment.py --exp_name $exp_name \
  --tracker_predictions_pkl tracker_downloads/Le3DE2E_tracking_predictions_val.pkl \
  --local_model_path "checkpoints/Qwen2.5-7B-Instruct" \
  --local_tokenizer_path "checkpoints/Qwen2.5-7B-Instruct" \
  --procs_per_task 128
```

#### ✅ Step 3: Retry If Needed

If any scenarios fail to generate or run, repeat this step with only the failed cases.

---

> 🔁 **Notes**:
> - Replace `$LLM` with the name of the LLM (e.g., `gemini`, `gpt4o`, `qwen`).
> - Replace `$exp_name` with your custom experiment name (e.g., `gemini_test_1`).
> - Ensure all required directories and files are correctly placed before running the pipeline.
