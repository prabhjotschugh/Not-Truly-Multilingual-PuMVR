# Not Truly Multilingual: Script Consistency as a Missing Dimension in VLM Evaluation
> **A parallel-script benchmark isolating orthography to evaluate true multilingual capability in Vision-Language Models.**
This repository contains the dataset, code, and evaluation results for the paper: **"Not Truly Multilingual: Script Consistency as a Missing Dimension in VLM Evaluation"** (PuMVR).

*This repository is fully anonymized for double-blind peer review.*

## 📖 Overview

Current Vision-Language Models (VLMs) are celebrated for their multilingual capabilities, yet they operate under a flawed assumption: treating orthography as a deterministic function of language. This overlooks billions of users of multi-script languages for whom a model's capability may be fractured by orthographic bias. 

We introduce **PuMVR** (Punjabi Multimodal Visual Reasoning), a parallel-script benchmark that isolates orthography as an independent variable. By evaluating 10 state-of-the-art VLMs across Gurmukhi, Shahmukhi, and Roman scripts, we expose a substantial and systematic **Script Gap**: models frequently solve visual tasks in one script while failing identical tasks in another. Our findings demonstrate that current "multilingual" VLMs are not truly multi-script.

## 📂 Repository Structure

```text
├── codes/                      # Inference and evaluation scripts for evaluated VLMs
│   ├── Claude Sonnet4/
│   ├── Google Gemini-2.5-Flash/
│   ├── Grok 4.1 Fast Reasoning/
│   ├── InternVL2_5-26B/
│   ├── Kimi-VL-A3B-Instruct/
│   ├── Llama-3.2-11B-Vision-Instruct/
│   ├── LLaVA-OneVision-1.5-8B-Instruct/
│   ├── OpenAI GPT-4o/
│   └── Qwen2-VL-7B-Instruct/
├── Full dataset/               # The PuMVR benchmark dataset
│   ├── dataset_images/         # Curated images for the visual reasoning tasks
│   └── dataset_json.json       # 1,000 parallel instances
└── Results/                    # Evaluation outputs and results analysis
    ├── iaa_results.csv         # Detailed evaluation metrics
    ├── exp1_results.xlsx       # Results for Experiment 1
    ├── exp2_results.xlsx       # Results for Experiment 2
    └── exp3_results.xlsx       # Results for Experiment 3
```

## 📊 The PuMVR Benchmark

The PuMVR dataset comprises **1,000 flat, strictly parallel instances** designed to isolate script-dependent bias.

Each instance contains:
- An image (AI-generated, public domain, or openly licensed)
- A question translated with strict semantic equivalence into **Gurmukhi**, **Shahmukhi**, and **Roman** scripts
- Four multiple-choice options per script
- The correct answer per script

**Annotation Quality:**
The dataset was verified by two independent native-speaking annotators with expertise across all three scripts. Agreement was measured using Prevalence-Adjusted Bias-Adjusted Kappa (PABAK), achieving scores of 0.970 or above for semantic equivalence and answer correctness, and 1.000 for script accuracy.

## 🎯 The Importance of Script Consistency Rate (SCR)

While traditional metrics evaluate per-script accuracy, they mask severe orthographic fragmentation. A model may achieve 90% accuracy in Gurmukhi and 85% in Shahmukhi, but fail to serve users reliably across both scripts.

To address this, we introduce the **Script Consistency Rate (SCR)**: the percentage of instances answered correctly across all three scripts simultaneously. 

SCR exposes the reality that high per-script performance does not guarantee script consistency. For a multi-script user, a model that succeeds in one script but fails the identical task in another is fundamentally unreliable. SCR should be a standard evaluation metric for any benchmark claiming multilingual capability to ensure equitable AI access.

## 🚀 Key Experiments & Findings

We evaluated 10 VLMs (e.g., GPT-4o, Gemini 2.5 Flash, Llama-3.2-11B-Vision, Qwen2-VL). Note: All experiments were conducted on a 375-instance evaluation split (from the 1000-instance dataset) to reserve data for future mitigation studies.

### 1. Script Gap Quantification
- **Objective:** Establish the existence and magnitude of script-dependent performance bias.
- **Design:** Generative evaluation across the 375 instances; models must output the exact text of the correct option. 
- **Findings:** Accuracy deltas between scripts reach up to 16%. SCR falls as low as 24.8%. 
- **Validation:** Manual verification of 11,250 responses confirmed 99.59% of errors were genuine comprehension failures (wrong-option selections), computationally validating the generative evaluation. Statistical significance of the gaps is confirmed via McNemar tests (with 95% CIs).

### 2. Modality Ablation
- **Objective:** Determine whether visual grounding compensates for script-specific weakness.
- **Design:** Text-only VLM condition vs. full multimodal condition.
- **Findings:** Visual Gain (VG) is approximately uniform across scripts per model. Images provide a *parallel benefit, not compensatory repair*; they boost absolute accuracy but do not shrink the orthographic gap. 

### 3. Cross-Script Few-Shot Transfer
- **Objective:** Test cross-script in-context knowledge transfer.
- **Design:** $k=3$ in-context examples provided in monoscript, cross-script, or mixed-script configurations.
- **Findings:** Few-shot knowledge transfer is highly brittle, exposing script-locked knowledge representations. Transfer Efficiency (TE) is asymmetric across scripts, and we observed severe in-context interference (Negative Few-Shot Lift) in low-resource script conditions.

## ⚠️ Limitations

- **Dataset Scope:** Experiments are run on a 375-instance split to prevent data snooping and reserve the remaining 625 instances for future studies. 
- **Generalizability:** The empirical findings are specific to Punjabi. While the parallel-script methodology is transferable to other multi-script languages, the specific performance behaviors are not automatically generalizable.
- **Evaluation Design:** We utilize a generative exact-text evaluation rather than a log-likelihood evaluation, validated by a 99.59% comprehension-error rate. 
- **Cultural Entanglement:** Some instances may carry subtle script-specific cultural associations (e.g., Gurmukhi/Sikh, Shahmukhi/Islamic contexts), which reflects the lived reality of these scripts but may introduce confounds beyond pure orthographic variation.

## 🛠️ Usage

### Prerequisites
*Details regarding dependencies and environment setup will be populated upon de-anonymization and final release.*

### Running Evaluations
Navigate to the respective model's directory within `codes/` and execute the provided Python scripts. For example, for GPT-4o:
```bash
cd codes/"OpenAI GPT-4o"/
python exp1.py
```
*Note: To run evaluations for `Qwen2-VL-72B-Instruct`, please use the scripts provided in the `Qwen2-VL-7B-Instruct` directory and simply update the model ID inside the code.*

## 🛑 Ethical Considerations
All images in PuMVR are AI-generated, public domain, or openly licensed. Human subjects depicted are from public historical archives or AI-generated to ensure there are no privacy concerns. The dataset contains no personally identifying information. This research exposes systemic bias to advocate for script-agnostic improvements that benefit marginalized communities.
