import os
import argparse
import json
import time
import numpy as np

import torch

from datasets import load_dataset, load_metric
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification, 
    EvalPrediction, 
    TrainingArguments, 
    Trainer,
    DataCollatorForTokenClassification,
    AutoModelForTokenClassification,
)

from utils import ( set_seed, NER_CLASSES, get_label_list, re_preprocessing, str2bool, )
from data_augmentation import DataAugmentationMethod
from mixup.utils import MixupAutoModelForSequenceClassification, MixupTrainer

parser = argparse.ArgumentParser()
parser.add_argument('--seed', default=0, type=int, help='Seed')
parser.add_argument('--model', default='bert-base', type=str, help='bert-base, roberta-large, roberta-base, roberta-small')
parser.add_argument('--task', default=0, type=int, help='TC, STS, NLI, NER, RE, DP, MRC, DST')
parser.add_argument('--output_dir', default='checkpoint/', type=str, help='Checkpoint directory/')
parser.add_argument('--result_dir', default='results/', type=str, help='Result directory/')
parser.add_argument('--lr', default=1e-5, type=float, help='Learning rate [1e-5, 2e-5, 3e-5, 5e-5]')
parser.add_argument('--wr', default=0., type=float, help='warm-up ratio [0., 0.1, 0.2, 0.6]')
parser.add_argument('--wd', default=0., type=float, help='weight decay coefficient [0.0, 0.01]')
parser.add_argument('--batch_size', default=8, type=int, help='batch size [8, 16, 32]')
parser.add_argument('--total_epochs', default=3, type=int, help='number of epochs [3, 4, 5, 10]')
parser.add_argument('--aug', default=False, type=str2bool, help="Do augmentation or not")
parser.add_argument('--aug_bt', default=True, type=str2bool, help="DA: Back translation")
parser.add_argument('--aug_rd', default=True, type=str2bool, help="DA: Random Swap")
parser.add_argument('--aug_rs', default=True, type=str2bool, help="DA: Random Deletion")
parser.add_argument('--mixup', default=False, type=str2bool, help="Mixup Method (LogitMix, MixOnGLUE)<-Should be modified manually in mixup/utils.py")

p_args = parser.parse_args()

start = time.time()

set_seed(p_args.seed)
if not os.path.exists(p_args.result_dir):
    os.makedirs(p_args.result_dir)

"""
TC(Topic Classification), STS(Semantic Textual Similarity), NLI(Natural Langauge Inference),
NER(Named Entity Recognition), RE(Relation Extraction), DP(Dependency Parsing),
MRC(Machine Reading Comprehension), DST(Dialogue State Tracking)
"""
KLUE_TASKS = ["ynat", "sts", "nli", "ner", "re", "dp", "mrc", "wos"]
KLUE_TASKS_REGRESSION = [False, True, False, False, False, ]
task_to_keys = {
    "ynat": ("title", None), # Macro F1 score
    "sts": ("sentence1", "sentence2"), # Pearson correlation coefficient, Macro F1 score
    "nli": ("premise", "hypothesis"), # Accuracy
    "ner": ("tokens", "ner_tags"), # Entity-level F1 score, Character-level F1 score
    "re": ("sentence", None), # Micro F1 score, AUPRC(averaged area under the precision recall curves)
    "dp": (), # ?
    "mrc": (), # ?
    "wos": (), # ?
}
sentence1_key, sentence2_key = task_to_keys[KLUE_TASKS[p_args.task]]
max_sequence_length = 512 if KLUE_TASKS[p_args.task] == "mrc" or KLUE_TASKS[p_args.task] == "wos" else 128
is_regression = KLUE_TASKS_REGRESSION[p_args.task]
label_column_name = "ner_tags" if p_args.task == 3 else "label"

def preprocess_function(examples):
    if p_args.task == 1:
        examples["label"] = [i["label"] for i in examples["labels"]]
        del examples["labels"]
    elif p_args.task == 4:
        examples[sentence1_key] = [re_preprocessing(examples["sentence"][i], examples["object_entity"][i], examples["subject_entity"][i]) for i in range(len(examples["sentence"]))]
    target = (
            (examples[sentence1_key],) if sentence2_key is None else (examples[sentence1_key], examples[sentence2_key])
        )
    
    result = tokenizer(*target, padding=True, max_length=max_sequence_length, truncation=True)
    
    """# Map labels to IDs (not necessary for GLUE tasks)
    if label_to_id is not None and "label" in examples:
        result["label"] = [(label_to_id[l] if l != -1 else -1) for l in examples["label"]]"""
    return result

def compute_metrics(p: EvalPrediction):
    if p_args.task == 3:
        predictions, labels = p
        predictions = np.argmax(predictions, axis=2)

        # Remove ignored index (special tokens)
        true_predictions = [
            [NER_CLASSES[label_list[p]] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        true_labels = [
            [NER_CLASSES[label_list[l]] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]

        results = metric.compute(predictions=true_predictions, references=true_labels)
        return results
    else:
        preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        preds = np.squeeze(preds) if is_regression else np.argmax(preds, axis=1)
        result = metric.compute(predictions=preds, references=p.label_ids)
        if len(result) > 1:
            result["combined_score"] = np.mean(list(result.values())).item()
        return result
    
# Load the dataset
#datasets = load_dataset("klue", KLUE_TASKS[p_args.task])
if p_args.aug:
    datasets = load_dataset("klue", KLUE_TASKS[p_args.task])
    # Save json file to local
    #datasets["train"].to_json(f"data/{KLUE_TASKS[p_args.task]}/train.json", orient="records", force_ascii=False, indent=2, lines=False, )
    #datasets["validation"].to_json(f"data/{KLUE_TASKS[p_args.task]}/validation.json", orient="records", force_ascii=False, indent=2, lines=False, )
    #exit()
    aug_data = DataAugmentationMethod(dataset=datasets, dataset_name=KLUE_TASKS[p_args.task],
     sentence1=sentence1_key, sentence2=sentence2_key, bt=p_args.aug_bt, eda=[False, False, p_args.aug_rs, p_args.aug_rd])
    aug_data.do_direct_aug()

    print("*** Data Augmentation is Finished. ***")

data_dir = f"data/{KLUE_TASKS[p_args.task]}"
data_files = {"train": [], "validation": []}
augment_list = [""]
if p_args.aug_bt:
    augment_list.append("_back_translation")
if p_args.aug_rd:
    augment_list.append("_random_deletion")
if p_args.aug_rs:
    augment_list.append("_random_swap")
for i in augment_list:     
    data_files["train"].append(f"{data_dir}{i}/train.json")
    data_files["validation"].append(f"{data_dir}{i}/validation.json")
datasets = load_dataset("json", data_dir=data_dir, data_files=data_files, field='data')

# Load the metric
metric = load_metric('./metric.py', KLUE_TASKS[p_args.task])

# Load the pre-trained model
label_list = []
if p_args.task != 1:
    label_list = get_label_list(datasets["train"][label_column_name], p_args.task)
num_labels = 1 if is_regression else len(label_list)
if p_args.task == 3:
    model = AutoModelForTokenClassification.from_pretrained(f"klue/{p_args.model}", num_labels=num_labels)
    label_to_id = {l: i for i, l in enumerate(label_list)}
else:
    if p_args.mixup:
        model = MixupAutoModelForSequenceClassification.from_pretrained(f"klue/{p_args.model}", num_labels=num_labels)
    else:
        model = AutoModelForSequenceClassification.from_pretrained(f"klue/{p_args.model}", num_labels=num_labels)

# Preprocessing the data
# Tokenize all texts and align the labels with them.
def tokenize_and_align_labels(examples):
    text_column_name = "tokens"
    label_column_name = "ner_tags"
    padding = "max_length"
    tokenized_inputs = tokenizer(
        examples[text_column_name],
        padding=padding,
        truncation=True,
        # We use this argument because the texts in our dataset are lists of words (with a label for each word).
        is_split_into_words=True,
    )
    labels = []
    for i, label in enumerate(examples[label_column_name]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            # Special tokens have a word id that is None. We set the label to -100 so they are automatically
            # ignored in the loss function.
            if word_idx is None:
                label_ids.append(-100)
            # We set the label for the first token of each word.
            elif word_idx != previous_word_idx:
                label_ids.append(label_to_id[label[word_idx]])
            # For the other tokens in a word, we set the label to either the current label or -100, depending on
            # the label_all_tokens flag.
            else:
                label_ids.append(label_to_id[label[word_idx]])
            previous_word_idx = word_idx

        labels.append(label_ids)
    tokenized_inputs["labels"] = labels
    return tokenized_inputs

tokenizer = AutoTokenizer.from_pretrained(f"klue/{p_args.model}")
if p_args.task != 3:
    preprocessed_datasets = datasets.map(preprocess_function, batched=True)
else:
    preprocessed_datasets = datasets.map(tokenize_and_align_labels, batched=True)
    data_collator = DataCollatorForTokenClassification(tokenizer)

# Learning rate [1e-5, 2e-5, 3e-5, 5e-5]
# warm-up ratio [0., 0.1, 0.2, 0.6]
# weight decay coefficient [0.0, 0.01]
# batch size [8, 16, 32]
# number of epochs [3, 4, 5, 10]
args = TrainingArguments(
    output_dir=p_args.output_dir,
    evaluation_strategy='epoch',
    learning_rate=p_args.lr,
    per_device_train_batch_size=p_args.batch_size,
    per_device_eval_batch_size=p_args.batch_size,
    num_train_epochs=p_args.total_epochs,
    weight_decay=p_args.wd,
    warmup_ratio=p_args.wr,
    seed=p_args.seed,
    save_total_limit=1,
)

if p_args.mixup:
    trainer = MixupTrainer(
        model,
        args,
        train_dataset=preprocessed_datasets["train"],
        eval_dataset=preprocessed_datasets["validation"],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )
else:
    trainer = Trainer(
        model,
        args,
        train_dataset=preprocessed_datasets["train"],
        eval_dataset=preprocessed_datasets["validation"],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

trainer.train()
test_result = trainer.evaluate()

elapsed_time = (time.time() - start) / 60 # Min.

path = f'{p_args.result_dir}model_{p_args.model}_lr_{p_args.lr}_wr_{p_args.wr}_wd_{p_args.wd}_bs_{p_args.batch_size}_te_{p_args.total_epochs}.json'
mode = 'a' if os.path.isfile(path) else 'w'

with open(path, mode) as f:
    if p_args.aug:
        result = {
            'seed': p_args.seed,
            'aug': {
                'bt': p_args.bt,
                'rd': p_args.rd,
                'rs': p_args.rs
            },
            f'{KLUE_TASKS[p_args.task]}': test_result,
            'time': elapsed_time,
        }
    else:
        result = {
            'seed': p_args.seed,
            f'{KLUE_TASKS[p_args.task]}': test_result,
            'time': elapsed_time,
        }
    json.dump(result, f, indent=2)
