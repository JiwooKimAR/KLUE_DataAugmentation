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
    AutoModelForQuestionAnswering,
    default_data_collator,
)

from utils import ( set_seed, NER_CLASSES, get_label_list, re_preprocessing, str2bool, postprocess_qa_predictions, )
from data_augmentation import DataAugmentationMethod
from mixup.utils import MixupAutoModelForSequenceClassification, MixupTrainer
from model.MRCTranier import QuestionAnsweringTrainer
from model.callbacks import Callbacks

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
parser.add_argument('--aug_bt', default=False, type=str2bool, help="DA: Back translation")
parser.add_argument('--aug_rd', default=False, type=str2bool, help="DA: Random Swap")
parser.add_argument('--aug_rs', default=False, type=str2bool, help="DA: Random Deletion")
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
KLUE_TASKS_REGRESSION = [False, True, False, False, False, False, False, False]
task_to_keys = {
    "ynat": ("title", None), # Macro F1 score
    "sts": ("sentence1", "sentence2"), # Pearson correlation coefficient, Macro F1 score
    "nli": ("premise", "hypothesis"), # Accuracy
    "ner": ("tokens", "ner_tags"), # Entity-level F1 score, Character-level F1 score
    "re": ("sentence", None), # Micro F1 score, AUPRC(averaged area under the precision recall curves)
    "dp": (), # ?
    "mrc": ("question" ,"context"), # ?
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
        if p_args.task != 6:
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
if p_args.task != 1 and p_args.task != 6:
    label_list = get_label_list(datasets["train"][label_column_name], p_args.task)
num_labels = 1 if is_regression else len(label_list)
if p_args.task == 3:
    model = AutoModelForTokenClassification.from_pretrained(f"klue/{p_args.model}", num_labels=num_labels)
    label_to_id = {l: i for i, l in enumerate(label_list)}
elif p_args.task == 6:
    model = AutoModelForQuestionAnswering.from_pretrained(f"klue/{p_args.model}")
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

def prepare_train_features(examples):
    # Preprocessing is slightly different for training and evaluation
    question_column_name = "question"
    context_column_name = "context"
    answer_column_name = "answers"

    # Padding side determines if we do (question|context) or (context|question)
    pad_on_right = tokenizer.padding_side == "right"
    
    tokenized_examples = tokenizer(
        examples[question_column_name if pad_on_right else context_column_name],
        examples[context_column_name if pad_on_right else question_column_name],
        truncation="only_second" if pad_on_right else "only_first",
        max_length=max_sequence_length,
        stride=128,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length"
    )

    sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
    offset_mapping = tokenized_examples.pop("offset_mapping")

    tokenized_examples["start_positions"] = []
    tokenized_examples["end_positions"] = []

    for i, offsets in enumerate(offset_mapping):
        # We will label impossible answers with the index of the CLS tokens
        input_ids = tokenized_examples["input_ids"][i]
        cls_index = input_ids.index(tokenizer.cls_token_id)

        # Grab the sequence corresponding to that example (to know what is the context and what is the question).
        sequence_ids = tokenized_examples.sequence_ids(i)

        # One example can give several spans, this is the index of the example containing this span of text.
        sample_index = sample_mapping[i]
        answers = examples[answer_column_name][sample_index]
        # If no answers are given, set the cls_index as answer.
        if len(answers["answer_start"]) == 0:
            tokenized_examples["start_positions"].append(cls_index)
            tokenized_examples["end_positions"].append(cls_index)
        else:
            # Start/end character index of the answer in the text.
            start_char = answers["answer_start"][0]
            end_char = start_char + len(answers["text"][0])

            # Start token index of the current span in the text.
            token_start_index = 0
            while sequence_ids[token_start_index] != (1 if pad_on_right else 0):
                token_start_index += 1

            # End token index of the current span in the text.
            token_end_index = len(input_ids) - 1
            while sequence_ids[token_end_index] != (1 if pad_on_right else 0):
                token_end_index -= 1

            # Detect if the answer is out of the span (in which case this feature is labeled with the CLS index).
            if not (offsets[token_start_index][0] <= start_char and offsets[token_end_index][1] >= end_char):
                tokenized_examples["start_positions"].append(cls_index)
                tokenized_examples["end_positions"].append(cls_index)
            else:
                # Otherwise move the token_start_index and token_end_index to the two ends of the answer.
                # Note: we could go after the last offset if the answer is the last word (edge case).
                while token_start_index < len(offsets) and offsets[token_start_index][0] <= start_char:
                    token_start_index += 1
                tokenized_examples["start_positions"].append(token_start_index - 1)
                while offsets[token_end_index][1] >= end_char:
                    token_end_index -= 1
                tokenized_examples["end_positions"].append(token_end_index + 1)

    return tokenized_examples

def prepare_validation_features(examples):
    # Preprocessing is slightly different for training and evaluation
    question_column_name = "question"
    context_column_name = "context"
    answer_column_name = "answers"

    # Padding side determines if we do (question|context) or (context|question)
    pad_on_right = tokenizer.padding_side == "right"

    # Tokenize our examples with truncation and maybe padding, but keep the overflows using a stride. This results
    # in one example possible giving several features when a context is long, each of those features having a
    # context that overlaps a bit the context of the previous feature.
    tokenized_examples = tokenizer(
        examples[question_column_name if pad_on_right else context_column_name],
        examples[context_column_name if pad_on_right else question_column_name],
        truncation="only_second" if pad_on_right else "only_first",
        max_length=max_sequence_length,
        stride=128,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length"
    )

    # Since one example might give us several features if it has a long context, we need a map from a feature to
    # its corresponding example. This key gives us just that.
    sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")

    # For evaluation, we will need to convert our predictions to substrings of the context, so we keep the
    # corresponding example_id and we will store the offset mappings.
    tokenized_examples["example_id"] = []

    for i in range(len(tokenized_examples["input_ids"])):
        # Grab the sequence corresponding to that example (to know what is the context and what is the question).
        sequence_ids = tokenized_examples.sequence_ids(i)
        context_index = 1 if pad_on_right else 0

        # One example can give several spans, this is the index of the example containing this span of text.
        sample_index = sample_mapping[i]
        tokenized_examples["example_id"].append(examples["guid"][sample_index])

        # Set to None the offset_mapping that are not part of the context so it's easy to determine if a token
        # position is part of the context or not.
        tokenized_examples["offset_mapping"][i] = [
            (o if sequence_ids[k] == context_index else None)
            for k, o in enumerate(tokenized_examples["offset_mapping"][i])
        ]

    return tokenized_examples

# Post-processing:
def post_processing_function(examples, features, predictions, stage="eval"):
    answer_column_name = "answers"

    # Post-processing: we match the start logits and end logits to answers in the original context.
    predictions = postprocess_qa_predictions(
        examples=examples,
        features=features,
        predictions=predictions,
        version_2_with_negative=False,
        n_best_size=20,
        max_answer_length=30,
        null_score_diff_threshold=0.0,
        output_dir="data/mrc/",
        prefix=stage,
    )
    # Format the result to the format the metric expects.
    formatted_predictions = [{"guid": k, "prediction_text": v} for k, v in predictions.items()]

    references = [{"guid": ex["guid"], "answers": ex[answer_column_name]} for ex in examples]
    return EvalPrediction(predictions=formatted_predictions, label_ids=references)

tokenizer = AutoTokenizer.from_pretrained(f"klue/{p_args.model}")
if p_args.task == 3:
    preprocessed_datasets = datasets.map(tokenize_and_align_labels, batched=True)
    data_collator = DataCollatorForTokenClassification(tokenizer)
elif p_args.task == 6:
    column_names = datasets["train"].column_names
    train_dataset = datasets["train"]
    train_dataset = train_dataset.map(prepare_train_features, batched=True, remove_columns=column_names)
    column_names = datasets["validation"].column_names
    validation_examples = datasets["validation"]
    validation_dataset = validation_examples.map(prepare_validation_features, batched=True, remove_columns=column_names)
    data_collator = (default_data_collator)
else:
    preprocessed_datasets = datasets.map(preprocess_function, batched=True)
    

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
        logging_strategy="no",
    )

if p_args.task == 3:
    trainer = Trainer(
        model,
        args,
        train_dataset=preprocessed_datasets["train"],
        eval_dataset=preprocessed_datasets["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )
elif p_args.task == 6:
    trainer = QuestionAnsweringTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        eval_examples=validation_examples,
        tokenizer=tokenizer,
        data_collator=data_collator,
        post_process_function=post_processing_function,
        compute_metrics=compute_metrics,
    )
else:
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

#trainer.add_callback(Callbacks)
trainer.train()
trainer.evaluate()

log_history = trainer.state.log_history

elapsed_time = (time.time() - start) / 60 # Min.

path = f'{p_args.result_dir}model_{p_args.model}_lr_{p_args.lr}_wr_{p_args.wr}_wd_{p_args.wd}_bs_{p_args.batch_size}_te_{p_args.total_epochs}.json'
mode = 'a' if os.path.isfile(path) else 'w'

with open(path, mode) as f:
    if p_args.aug:
        result = {
            'seed': p_args.seed,
            'aug': {
                'bt': p_args.aug_bt,
                'rd': p_args.aug_rd,
                'rs': p_args.aug_rs
            },
            f'{KLUE_TASKS[p_args.task]}': log_history,
            'time': elapsed_time,
        }
    else:
        result = {
            'seed': p_args.seed,
            f'{KLUE_TASKS[p_args.task]}': log_history,
            'time': elapsed_time,
        }
    json.dump(result, f, indent=2)
