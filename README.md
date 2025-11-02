# Identifying Concurrency Bug Reports via Linguistic Patterns

This repository provides the resources and instructions for reproducing the experiments in our paper  
**"Identifying Concurrency Bug Reports via Linguistic Patterns"** (submitted to *Empirical Software Engineering*, 2025).

---

## 📦 Datasets and Trained Models

Both the dataset and the fine-tuned model used in our experiments are publicly available on **Zenodo**.

### 🗂️ Dataset
Includes **10,920 bug reports** (from GitHub and Jira) in `.txt` format, annotated for concurrency-related content.  
Each file contains one bug report and its corresponding label (`concurrency_bug = 1/0`).

📥 **Download:** [https://zenodo.org/records/17490237](https://zenodo.org/records/17490237)

### ⚙️ Fine-tuned Model (CTagger)
The best-performing configuration **(CodeBERT + ALL linguistic patterns)**,  
fine-tuned on `Dataset_Git` and validated on `Dataset_Jira` and `Dataset_Post`.

📥 **Download:** [https://zenodo.org/records/17490237](https://zenodo.org/records/17490237)

---

## 🧪 Reproducing the Results

This section explains how to reproduce our experiments for fine-tuning PLMs and running CTagger (our best-performing model).

---

### 🔧 1. Prepare the Data

Download and unzip the dataset and extracted linguistic patterns from Zenodo:

```bash
unzip dataset.zip
unzip patterns.zip
```

#### 📦 Files included

- `dataset/` — 10,920 bug reports (GitHub, Jira, and Post-cutoff)
- `patterns/` — extracted linguistic patterns at four levels (KW, PH, SE, BR)

---

### ⚙️ 2. Process the Data

Use the provided script to merge bug reports with their corresponding linguistic patterns.

```bash
cd plm

# Example: generate LP-enhanced training and testing sets
python process.py \
  --br_dir ../dataset/training/ \
  --patterns_dir ../patterns/training/ \
  --output_dir ../lps/training/

# For specific datasets (e.g., GitHub, Jira, or Post-cutoff)
python process.py \
  --br_dir ../dataset/github/ \
  --patterns_dir ../patterns/github/ \
  --output_dir ../lps/github/
```

**💡 Tip:**  
You can specify a specific pattern combination using `--pattern`, e.g., `--pattern KW+PH`, or `--pattern ALL` for all patterns.

---

### 🧠 3. Fine-tune the PLMs

Once the LP-augmented datasets are prepared, fine-tune the selected model:

```bash
python fine_tune.py \
  --model codebert \
  --train_dir ../lps/training/ \
  --test_dir ../lps/github/
```

**Supported models:** `bert`, `roberta`, `albert`, `codebert`, `graphcb`

---

### 🚀 4. Run CTagger (Our Best PLM)

Unzip the pretrained CTagger model (CodeBERT+ALL) and run it for evaluation:

```bash
unzip lp_model.zip

# Prepare ALL-pattern test data
python process.py \
  --br_dir ../dataset/github/ \
  --patterns_dir ../patterns/github/ \
  --output_dir ../lps/github/ \
  --pattern ALL

# Run inference with CTagger
python plm.py \
  --model_dir ../lp_model/ \
  --data_root ../lps/github/ \
  --pattern ALL
```

---