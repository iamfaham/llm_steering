# Prompt Steering vs Activation Steering in LLMs

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1zgN3ydePd4NqPxRQQ7DKRyCc5NikBMIQ?usp=sharing)

A controlled experimental comparison of prompt-based behavioral steering and activation steering in transformer language models.

This project investigates whether activation steering provides more stable behavioral control than prompt engineering — particularly under prompt conflict.

---

## Motivation

Large Language Models (LLMs) are commonly steered using prompts:

> "You are a confident and decisive expert."

However, prompt steering operates at the input level and competes with the model’s internal representations.

This project explores an alternative:

> **Activation Steering** — directly modifying hidden states at inference time without changing model weights.

We test whether activation steering produces more stable behavioral control compared to prompt steering.

---

## Research Question

Does activation steering provide more consistent control over model confidence than prompt steering, especially under conflicting instructions?

We evaluate behavior along one axis:

**Confidence vs Hedging**

---

## Experimental Design

We compare three generation modes using the same frozen model:

1. **Baseline** — No steering  
2. **Prompt-Steered** — Explicit confidence instructions  
3. **Activation-Steered** — Hidden-state modification via steering vector  

All comparisons use identical decoding parameters.

---

## Model

- `google/gemma-3-4b-it`
- Loaded via Hugging Face Transformers
- No fine-tuning or weight updates
- Steering performed via forward hooks

---

## Activation Steering Method

We construct a steering vector by contrasting hidden states from:

- Confident prompts
- Hedging prompts

The steering vector is defined as:
```
mean(hidden_confident) - mean(hidden_hedging)
```

This vector is injected into a selected transformer layer during generation:
```
modified_hidden = hidden + α * steering_vector
```


Where:
- `α` controls steering strength
- Layer choice controls where behavior is influenced

---

## Evaluation Metrics

We compute a composite **Confidence Score**:
```
confidence_score = assertive_markers - hedge_markers - contrast_markers
```


Evaluation includes:

- Mean ± standard deviation over multiple runs
- Stress testing with conflicting prompt instructions
- Layer ablation
- Alpha sweep
- Coherence inspection

---

## Key Findings

### Prompt Steering Is Brittle

Under stress conditions:

- Confidence score decreases
- Variance increases
- Model reverts to balanced hedging

---

### Activation Steering Is More Stable

Across multiple generations:

- Similar or higher mean confidence
- Lower variance
- Better resistance to prompt conflict

---

### Steering–Coherence Tradeoff

Early-layer steering (e.g., Layer 6):

- Produces extremely high confidence scores
- Causes repetition and structural degeneration

Middle layers provide better balance between:
- Behavioral control
- Output coherence

---

## Example Results

Stress Test (Mean ± Std):
```
Prompt-Steered: 1.4 ± 1.36
Activation-Steered: 2.0 ± 1.10
```

This demonstrates improved stability under prompt conflict.

---

## Limitations

- Effects are modest in instruction-tuned models (RLHF alignment dampens shifts)
- Steering strength must be carefully tuned
- Early-layer steering can degrade coherence
- Confidence metric is heuristic-based

Activation steering is not a replacement for fine-tuning — it is an inference-time control technique.

---

## What This Project Demonstrates

This project shows that:

- Activation steering can produce measurable internal bias shifts
- Behavioral control can be achieved without retraining
- There is a clear tradeoff between steering strength and coherence
- Prompt steering and activation steering behave differently under stress

---

## Requirements

```bash
pip install transformers accelerate torch matplotlib
```

GPU recommended (tested in Google Colab with Tesla T4 GPU).

---

## Future Extensions

- Compare base vs instruction-tuned models
- Add repetition and coherence metrics
- Explore other behavioral axes (e.g., risk aversion, moral caution)
- Evaluate steering transfer across prompts
