# Roleplay Jailbreak Evaluation and Defense in Open-Source LLMs

AAU Master's Thesis Project · Semester 9 · 2025

Evaluates the susceptibility of open-source LLMs to roleplay-based jailbreak attacks and measures the effectiveness of response-level defense using Qwen3Guard. Six models are tested on both base and persona-style AdvBench prompts, with LlamaGuard-3 as the safety judge.

## Models Evaluated
| Model | Parameters |
|-------|-----------|
| Falcon-11B | 11B |
| Falcon-7B-Instruct | 7B |
| Mistral-7B-Instruct | 7B |
| Llama-2-7B-Chat | 7B |
| Llama-2-13B-Chat | 13B |
| Phi-3-mini | 3.8B |

## Pipeline
```
AdvBench (520 prompts)
    ↓ roleplay transformation (3 personas × 520 = 1,560 prompts)
    ↓ Target LLM generates response
    ↓ LlamaGuard-3 judges: safe / unsafe   (ASR before defense)
    ↓ Qwen3Guard filters: block / allow    (ASR after defense)
```

## Key Results
| Model | ASR Base | ASR Roleplay | ASR + Qwen3Guard |
|-------|----------|--------------|------------------|
| Falcon-11B | 81.2% | 90.9% | 0.8% |
| Falcon-7B | 36.5% | 86.9% | 1.3% |
| Mistral-7B | 71.2% | 81.8% | 1.5% |
| Llama-2-7B | 25.6% | 28.7% | 0.5% |
| Llama-2-13B | 3.8% | 4.1% | 0.2% |
| Phi-3-mini | 22.8% | 23.1% | 3.3% |

## Setup
```bash
conda create -n jailbreak python=3.10 -y
conda activate jailbreak
pip install transformers accelerate bitsandbytes pandas
```

## Dataset
[AdvBench](https://arxiv.org/abs/2307.15043) — 520 harmful behaviors (Zou et al. 2023)
