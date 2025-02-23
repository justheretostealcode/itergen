# IterGen
Iterate on LLM generation using grammar


## SQL Case Study

To run all SQL experiments, use the following command:
```bash
python3 case_studies/sql/eval_sql.py
```

### Results
The results of the experiments will be stored in the `results/sql_results/results.jsonl` file. Each line in this file contains detailed information about the performance of a model on different SQL tasks.

Field Explanation Table:

| Field | Description |
| --- | --- |
| `model_id` | Model id used for the experiment |
| `evaluation_mode` | Evaluation mode used for the experiment |
| `temperature` | Temperature used for the experiment |
| `recurrence_penalty` | Recurrence penalty used for the experiment |
| `seed` | Seed used for the experiment |
| `execution_accuracy` | Accuracy of the model on different difficulty levels (easy, medium, hard, extra, all) |
| `Counts` | Number of problems in each difficulty level |
| `error_types` | Error types of the model |
| `results_jsonl_file` | Path to the result jsonl file |
| `avg_num_tokens` | Average number of tokens in the generated code |
| `num_of_problems` | Number of problems in the experiment |
| `Average time` | Average time taken to generate the code |

An example of a result entry looks like this:
```javascript
{
    'model_id': 'model_id', 
    'evaluation_mode': 'itergen', 
    'temperature': None, 
    'recurrence_penalty': 0.3, 
    'seed': 0, 
    'execution_accuracy': [
        ('easy', 0.5), 
        ('medium', 0.667), 
        ('hard', 0), 
        ('extra', 0), 
        ('all', 0.6)
    ], 
    'Counts': [
        ('easy', 4), 
        ('medium', 6), 
        ('hard', 0), 
        ('extra', 0), 
        ('all', 10)
    ], 
    'error_types': {
        'Valid': 10
    }, 
    'results_jsonl_file': 'results/sql_results/model_id/itergen_tempt:None_seed:0_rp:0.3_maxiter_20_num:10.jsonl', 
    'avg_num_tokens': 26.6, 
    'num_of_problems': 10, 
    'Average time': 1.313
}
```

### Comparison script

To run the script to compare two result jsonl file the command line, use the following format:

```bash
python case_studies/sql/compare.py --file1 <path_to_file1> --file2 <path_to_file2>
```

## Privacy Leakage Case Study

To run a single privacy case study, use the following command:
```bash
python3 case_studies/privacy/privacy_evaluation.py
```

To run all the experiments from section 4.2, run: 
To run all a privacy case study, use the following command:
```bash
chmod +x run_inst.sh
./run_inst.sh
```

### Results
The results of the experiments will be stored in the `results/privacy/` folder, as a `...completions.json` file containing generation data, along with `...results.jsonl` file, containing evaluation metrics. 

Field Explanation Table:

| Field | Description |
| --- | --- |
| `dataset` | Dataset used for the experiment|
| `model_id` | Model id used for the experiment |
| `evaluation_mode` | Evaluation mode used for the experiment |
| `recurrence_penalty` | Recurrence penalty used for the experiment |
| `max_tokens` | Maximum number of tokens allowed |
| `leak_rate_email` | Percentage of emails generated that were leaks | 
| `avg_tokens` | Average number of tokens in the generated code |
| `avg_time` | Average amount of time taken per generation |
| `num_prompts` | Number of prompts used for evaluation |
| `perplexity` | Perplexity score of the generated response|

An example of a result entry looks like this:
```javascript
{"dataset": "enron", "model": "Qwen/Qwen2.5-0.5B", "recurrence_penalty": 0.3, "max_tokens": 19, "leak_rate_email": 0.0, "avg_tokens": 24.14, "avg_time": 0.46, "num_prompts": 100, "mode": "itergen", "perplexity": 6.314}
```
