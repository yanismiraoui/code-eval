# CodeEval Dashboard 📊

A dashboard for comparing DeepSeek-Coder and CodeGemma models on various code benchmarks using Pass@k metrics.

[Dashboard](https://code-eval.replit.app/)

*Note: The evaluation results are limited due to lack of access to high-end GPUs.*

## Overview

### Running the evaluations 🧪
This project provides a framework for evaluating and comparing the performance of DeepSeek-Coder and CodeGemma models on three benchmarks:
- HumanEval
- MBPP (Mostly Basic Programming Problems)
- Multipl-E (Running without Docker with necessary dependencies installed)

The evaluation uses the Pass@k metric for k=1, 3, and 5.

The models used by default are the smallest versions of DeepSeek-Coder and CodeGemma from Hugging Face (running on Mac M1 GPU):
- DeepSeek-Coder 1.3B base model: [link](https://huggingface.co/deepseek-ai/deepseek-coder-1.3b-base)
- CodeGemma 2B model: [link](https://huggingface.co/google/codegemma-2b)

*Note: You need a Hugging Face token to download the models as well as accept the license agreements of the models.*

### Running the dashboard and viewing the results 🚀

The dashboard is built with Streamlit and allows you to view the results of the evaluations. A version of the dashboard is hosted via Replit [here](https://code-eval.replit.app/).

You can also run the dashboard locally by running `streamlit run src/dashboard/app.py`.

## Setup

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yanismiraoui/CodeEval.git
```

2. Navigate to the project directory:
```bash
cd CodeEval
```

3. Create conda environment:
```bash
conda env create -f environment.yml
```

4. Activate the environment:
```bash
conda activate code-eval
```

### Configuration

1. Copy the environment template:
```bash
cp .env.template .env
```

2. Edit the .env file and add your Hugging Face token:
```bash
# Get your token from https://huggingface.co/settings/tokens
HF_TOKEN=your_token_here
```

## Usage

1. Run the sweep:
```bash
python scripts/run_sweep.py
```

2. Run a single benchmark and model, for example, MBPP (50 problems) with DeepSeek-Coder and k=3:
```bash
python scripts/run_single_eval.py --benchmark mbpp --model deepseek --k 3 --num_problems 50 --temperature 0.2 --max_length 512
```

3. View the results in the dashboard (using the results saved in `streamlit_data/`):
```bash
streamlit run src/dashboard/app.py
```

## Repo structure 📁

- `scripts/`: Scripts for running the evaluations and sweeps.
- `src/evaluator/`: Contains the evaluator classes for each benchmark.
- `src/utils/`: Utility functions for the project.
- `src/config/`: Configuration for the project defaults.
- `.env`: Environment variables for the project (Hugging Face token).
- `src/dashboard/`: The Streamlit dashboard for visualizing the results.
- `tests/`: Tests for the project.
- `environment.yml`: The conda environment file.


## Limitations 🚧
The evaluation is currently limited to 1 sample of k completions per problem because of the computational cost of running multiple samples (e.g. if k=5 and num_samples=5, then 25 completions are needed for each problem). In other words, the pass@k metric could be better estimated by running more samples and this is a parameter that is available but has not been used to obtain the results.