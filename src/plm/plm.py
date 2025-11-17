import argparse
import os
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import Dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


def load_data_from_folder(folder, label, data_list, file_names=None, pattern=None):
    if not os.path.exists(folder):
        print(f"⚠️ Warning: folder not found: {folder}")
        return
    for filename in os.listdir(folder):
        if filename.endswith(".txt"):
            filepath = os.path.join(folder, filename)
            with open(filepath, 'r', encoding='utf-8') as file:
                report_text = file.read()
            data_list.append({"text": report_text, "label": label})
            if file_names is not None:
                file_names.append(f"{pattern}/{filename}")


def compute_metrics(labels, preds):
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    acc = accuracy_score(labels, preds)
    return {'accuracy': acc, 'precision': precision, 'recall': recall, 'f1': f1}


def main():
    parser = argparse.ArgumentParser(description="Evaluate CTagger model on multiple linguistic patterns.")
    parser.add_argument("--model_dir", type=str, required=True, help="Path to the fine-tuned model directory (e.g., ./codebert_model)")
    parser.add_argument("--data_root", type=str, required=True, help="Root folder of the pattern-based test data (e.g., ./lps/)")
    parser.add_argument("--pattern", type=str, required=True, help="Give a linguistic pattern to evaluate.")
    parser.add_argument("--output_dir", type=str, default="Results", help="Output directory for evaluation results.")
    parser.add_argument("--max_length", type=int, default=512, help="Maximum token length for truncation.")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"\n🚀 Loading model from {args.model_dir}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_dir).to(device)
    model.eval()
    pattern = args.pattern
    print(f"\n==============================")
    print(f"🔍 Evaluating pattern: {pattern}")
    print(f"==============================")

    test_data, test_file_names = [], []
    con_path = os.path.join(args.data_root, pattern, "con")
    non_path = os.path.join(args.data_root, pattern, "non")

    load_data_from_folder(con_path, 1, test_data, test_file_names, pattern)
    load_data_from_folder(non_path, 0, test_data, test_file_names, pattern)

    print(f"✅ Loaded {len(test_data)} test samples for {pattern}")

    test_dataset = Dataset.from_list(test_data)

    def tokenize_function(example):
        return tokenizer(example['text'], truncation=True, padding='max_length', max_length=args.max_length)

    tokenized_test_dataset = test_dataset.map(tokenize_function, batched=True)
    tokenized_test_dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])

    predicted_labels, true_labels = [], []
    with torch.no_grad():
        for i in range(len(tokenized_test_dataset)):
            inputs = {
                "input_ids": tokenized_test_dataset[i]["input_ids"].unsqueeze(0).to(device),
                "attention_mask": tokenized_test_dataset[i]["attention_mask"].unsqueeze(0).to(device),
            }
            outputs = model(**inputs)
            predicted_label = torch.argmax(outputs.logits, dim=-1).item()
            predicted_labels.append(predicted_label)
            true_labels.append(tokenized_test_dataset[i]["label"])

    metrics = compute_metrics(true_labels, predicted_labels)
    print("\n=== Evaluation Metrics ===")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")

    output_file_path = os.path.join(args.output_dir, f"lps_{pattern}.txt")
    with open(output_file_path, "w", encoding="utf-8") as output_file:
        output_file.write(f"# pattern: {pattern}\n")
        for file_name, label in zip(test_file_names, predicted_labels):
            output_file.write(f"{file_name}: {label}\n")

    print(f"✅ Results saved to {output_file_path}")


if __name__ == "__main__":
    main()
