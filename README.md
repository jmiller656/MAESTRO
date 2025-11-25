# MAESTRO

MAESTRO: Method for Agent Enhancement through Standardized Tool Reliability Optimization.

This Repo and code is forked from [Workbench](https://github.com/olly-styles/WorkBench.git)

## Installation

Python Version: 3.10.11

```bash
pip install -r requirements.txt
```

## Usage

All five sandbox databases, task-outcome pairs, and pre-computed inference results are provided in the `data` directory.

### Evaluation

As all the pre-computed inference results are provided, you can reproduce the evaluation results in the paper without running inference. Use the following script to calculate the metrics.

```bash
python scripts/evals/calculate_all_metrics.py;
python scripts/evals/calculate_all_metrics.py --all_tools;
```

Note that results are not provided for the all_tools variant of GPT3.5 and LLama2-70B as the prompt does not fit into the context window for these models. 

### Data generation
All generated data is provided pre-computed in the `data` directory. If you want to generate the data yourself, follow the steps below.

```bash
python scripts/data_generation/mocked_data/generate_all_mocked_data.py;
python scripts/data_generation/query_answer_generation/generate_all_query_and_answer.py;
```


### Inference

#### Run inference for specific domain and model
```bash
python scripts/inference/generate_results.py --model_name MODEL_NAME --queries_path QUERIES_PATH
```

#### Run inference for all domains and models 
```bash
python scripts/inference/generate_all_results.py
```


