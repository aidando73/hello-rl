Baseline from https://github.com/TrelisResearch/ADVANCED-fine-tuning
```bash
python3 sampling_batch_inference.py --dataset gsm8k --model "unsloth/Llama-3.1-8B-Instruct" --num-samples 8 --gsm8k-file gsm8k_test_1319.json
```

Metrics:
Total questions attempted: 1319
Pass@8 (at least 1 correct): 1273/1319 (96.51%)
Majority@8 (majority correct): 1133/1319 (85.90%)
