import os
import json
import torch
import argparse
import random
import itertools
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, DataCollatorWithPadding
from datasets import Dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def parse_arguments():
    parser = argparse.ArgumentParser(description="Fine-tune a PLM on bug report classification")
    parser.add_argument("--model", type=str, required=True, choices=["albert", "codebert", "graphcb", "roberta", "bert"])
    parser.add_argument("--train_dir", type=str, required=True, help="Path to the training data directory.")
    parser.add_argument("--test_dir", type=str, required=True, help="Path to the testing data directory.")
    parser.add_argument("--batch", type=int, default=32, help="Batch size for training and evaluation")
    return parser.parse_args()


def set_random_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_model_name(model_key):
    model_dict = {
        'albert': 'albert-base-v2',
        'codebert': 'microsoft/codebert-base',
        'graphcb': 'microsoft/graphcodebert-base',
        'roberta': 'roberta-base',
        'bert': 'bert-base-cased'
    }
    return model_dict.get(model_key, None)


def load_data_from_folder(folder, label, data_list, file_names=None):
    for filename in os.listdir(folder):
        if filename.endswith(".txt"):
            filepath = os.path.join(folder, filename)
            with open(filepath, 'r', encoding='utf-8') as file:
                report_text = file.read()
                data_list.append({"text": report_text, "label": label})
                if file_names is not None:
                    file_names.append(filename)


def prepare_datasets(train_paths, test_paths):
    train_data, test_data, test_file_names = [], [], []

    load_data_from_folder(train_paths[0], 1, train_data)
    load_data_from_folder(train_paths[1], 0, train_data)
    load_data_from_folder(test_paths[0], 1, test_data, test_file_names)
    load_data_from_folder(test_paths[1], 0, test_data, test_file_names)

    return Dataset.from_list(train_data), Dataset.from_list(test_data)


def tokenize_function(example, tokenizer):
    return tokenizer(example['text'], truncation=True, padding='max_length', max_length=512)


def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }


def save_best_model(eval_metrics, best_model_dir, pattern_name, model_name):
    os.makedirs(best_model_dir, exist_ok=True)

    # model.save_pretrained(best_model_dir)
    # tokenizer.save_pretrained(best_model_dir)

    with open(f"result_{model_name}.txt", "a") as f:
        f.write(f"Pattern: {pattern_name}\n")
        f.write(f"Precision: {eval_metrics.get('eval_precision', 0):.4f}\n")
        f.write(f"Recall: {eval_metrics.get('eval_recall', 0):.4f}\n")
        f.write(f"F1 Score: {eval_metrics.get('eval_f1', 0):.4f}\n")
        f.write("=" * 40 + "\n")

    print(f"✅ Saved model and evaluation metrics for pattern [{pattern_name}]")


def main():
    args = parse_arguments()
    set_random_seed()
    device = get_device()
    model_name = get_model_name(args.model)

    if model_name is None:
        raise ValueError(f"Unknown model: {args.model}. Please provide a valid model name.")

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    patterns = ["KW", "PH", "SE", "BR", "KW+PH", "KW+SE", "KW+BR", "PH+SE", "PH+BR", "SE+BR", "KW+PH+SE", "KW+PH+BR", "KW+SE+BR", "PH+SE+BR", "ALL"]
    for pattern in patterns:
        print(f"\n🚀 Training for pattern combination: {pattern}\n")

        train_paths = [f"{args.train_dir}/{pattern}/con/", f"{args.train_dir}/{pattern}/non/"]
        test_paths = [f"{args.test_dir}/{pattern}/con/", f"{args.test_dir}/{pattern}/non/"]

        train_dataset, test_dataset = prepare_datasets(train_paths, test_paths)
        tokenized_train_dataset = train_dataset.map(lambda x: tokenize_function(x, tokenizer), batched=True)
        tokenized_test_dataset = test_dataset.map(lambda x: tokenize_function(x, tokenizer), batched=True)
        tokenized_train_dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])
        tokenized_test_dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])

        output_dir = f"./results/{pattern}"
        os.makedirs(output_dir, exist_ok=True)

        training_args = TrainingArguments(
            output_dir=output_dir,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            save_total_limit=1,
            learning_rate=2e-5,
            per_device_train_batch_size=args.batch,
            per_device_eval_batch_size=args.batch,
            num_train_epochs=3,
            weight_decay=0.01,
            load_best_model_at_end=True,
            metric_for_best_model="eval_f1",
            greater_is_better=True,
        )

        trainer = Trainer(
            model=AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2).to(device),
            args=training_args,
            train_dataset=tokenized_train_dataset,
            eval_dataset=tokenized_test_dataset,
            tokenizer=tokenizer,
            compute_metrics=compute_metrics,
        )

        trainer.train()
        eval_results = trainer.evaluate()
        save_best_model(eval_results, f"./temp/{pattern}", pattern, args.model)

    print("\n✅ All pattern combinations have been trained!\n")


if __name__ == "__main__":
    main()
