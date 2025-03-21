Baseline from https://github.com/TrelisResearch/ADVANCED-fine-tuning
```bash
python3 sampling_batch_inference.py --dataset gsm8k --model "unsloth/Llama-3.1-8B-Instruct" --num-samples 8 --gsm8k-file gsm8k_test_1319.json
```

### Baseline
- Pass@8 (at least 1 correct): 1273/1319 (96.51%)
- Majority@8 (majority correct): 1133/1319 (85.90%)

### 1050 steps
- Pass@8 (at least 1 correct): 1154/1319 (87.49%)
- Majority@8 (majority correct): 1130/1319 (85.67%)

### 1950 steps
- Pass@8 (at least 1 correct): 1178/1319 (88.70%)
- Majority@8 (majority correct): 1156/1319 (86.58%)
- maj@8 has increased from baseline and pass@8 increased from 1050 steps

### 3300 steps
- Pass@8 (at least 1 correct): 1277/1319 (96.82%)
- Majority@8 (majority correct): 1272/1319 (96.44%)
- I beleive results are skewed due to LLM as a judge hallucinations

### 4 bit merged model - baseline
- Pass@8 (at least 1 correct): 1143/1319 (86.66%)
- Majority@8 (majority correct): 1118/1319 (84.76%)

### 4 bit merged model after 250 steps
- Pass@8 (at least 1 correct): 1140/1319 (86.43%)
- Majority@8 (majority correct): 1117/1319 (84.69%)

## Cost
- 61 hrs of training total
- On an 1 x H100 SXM - 48 vCPU 188 GB RAM - $2.99/hr USD
- $182.39 total

## Big Math RL

## Qwen 2.5 3B Instruct

### Baseline
- Pass@8 (at least 1 correct): 1167/1319 (88.48%)
- Majority@8 (majority correct): 1128/1319 (85.52%)

### 4bit merged model - baseline
- Pass@8 (at least 1 correct): 1186/1319 (89.92%)
- Majority@8 (majority correct): 1091/1319 (82.71%)


### Summarization training costs
- $6.16/hr
- 47 hrs total ran
- $289.52 total