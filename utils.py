import random
import argparse
import numpy as np
import torch

import datasets


NER_CLASSES = np.array(["B-PS", "I-PS", "B-LS", "I-LC", "B-OG", "I-OG", "B-DT", "I-DT", "B-TI", "I-IT", "B-QT", "I-QT", "O"])

DATASET_FEATURES = [{
                "guid": datasets.Value("string"),
                "title": datasets.Value("string"),
                "label": datasets.features.ClassLabel(names=["IT과학", "경제", "사회", "생활문화", "세계", "스포츠", "정치"]),
                "url": datasets.Value("string"),
                "date": datasets.Value("string"),
            },
            {
                "guid": datasets.Value("string"),
                "source": datasets.Value("string"),
                "sentence1": datasets.Value("string"),
                "sentence2": datasets.Value("string"),
                "labels": {
                    "label": datasets.Value("float64"),
                    "real-label": datasets.Value("float64"),
                    "binary-label": datasets.ClassLabel(names=["negative", "positive"]),
                },
            },
            {
                "guid": datasets.Value("string"),
                "source": datasets.Value("string"),
                "premise": datasets.Value("string"),
                "hypothesis": datasets.Value("string"),
                "label": datasets.ClassLabel(names=["entailment", "neutral", "contradiction"]),
            },
            {
                "sentence": datasets.Value("string"),
                "tokens": datasets.Sequence(datasets.Value("string")),
                "ner_tags": datasets.Sequence(
                    datasets.ClassLabel(
                        names=[
                            "B-DT",
                            "I-DT",
                            "B-LC",
                            "I-LC",
                            "B-OG",
                            "I-OG",
                            "B-PS",
                            "I-PS",
                            "B-QT",
                            "I-QT",
                            "B-TI",
                            "I-TI",
                            "O",
                        ]
                    )
                ),
            },
            {
                "guid": datasets.Value("string"),
                "sentence": datasets.Value("string"),
                "subject_entity": {
                    "word": datasets.Value("string"),
                    "start_idx": datasets.Value("int32"),
                    "end_idx": datasets.Value("int32"),
                    "type": datasets.Value("string"),
                },
                "object_entity": {
                    "word": datasets.Value("string"),
                    "start_idx": datasets.Value("int32"),
                    "end_idx": datasets.Value("int32"),
                    "type": datasets.Value("string"),
                },
                "label": datasets.ClassLabel(
                    names=[
                        "no_relation",
                        "org:dissolved",
                        "org:founded",
                        "org:place_of_headquarters",
                        "org:alternate_names",
                        "org:member_of",
                        "org:members",
                        "org:political/religious_affiliation",
                        "org:product",
                        "org:founded_by",
                        "org:top_members/employees",
                        "org:number_of_employees/members",
                        "per:date_of_birth",
                        "per:date_of_death",
                        "per:place_of_birth",
                        "per:place_of_death",
                        "per:place_of_residence",
                        "per:origin",
                        "per:employee_of",
                        "per:schools_attended",
                        "per:alternate_names",
                        "per:parents",
                        "per:children",
                        "per:siblings",
                        "per:spouse",
                        "per:other_family",
                        "per:colleagues",
                        "per:product",
                        "per:religion",
                        "per:title",
                    ]
                ),
                "source": datasets.Value("string"),
            },
            {
                "sentence": datasets.Value("string"),
                "index": [datasets.Value("int32")],
                "word_form": [datasets.Value("string")],
                "lemma": [datasets.Value("string")],
                "pos": [datasets.Value("string")],
                "head": [datasets.Value("int32")],
                "deprel": [datasets.Value("string")],
            },
            {
                "title": datasets.Value("string"),
                "context": datasets.Value("string"),
                "news_category": datasets.Value("string"),
                "source": datasets.Value("string"),
                "guid": datasets.Value("string"),
                "is_impossible": datasets.Value("bool"),
                "question_type": datasets.Value("int32"),
                "question": datasets.Value("string"),
                "answers": datasets.features.Sequence(
                    {
                        "answer_start": datasets.Value("int32"),
                        "text": datasets.Value("string"),
                    },
                ),
            },
            {
                "guid": datasets.Value("string"),
                "domains": [datasets.Value("string")],
                "dialogue": [
                    {
                        "role": datasets.Value("string"),
                        "text": datasets.Value("string"),
                        "state": [datasets.Value("string")],
                    }
                ],
            }, ]


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # Multi GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False
    # ^^ safe to call this function even if cuda is not available


def re_preprocessing(sentence, object_entity, subject_entity):
    if object_entity["start_idx"] < subject_entity["start_idx"]:
        sentence = sentence[:object_entity["start_idx"]] + "<obj>" + object_entity["word"] + "</obj>" + sentence[object_entity["end_idx"]+1:]
        sentence = sentence[:subject_entity["start_idx"]+11] + "<subj>" + subject_entity["word"] + "</subj>" + sentence[subject_entity["end_idx"]+12:]
    else:
        sentence = sentence[:subject_entity["start_idx"]] + "<subj>" + subject_entity["word"] + "</subj>" + sentence[subject_entity["end_idx"]+1:]
        sentence = sentence[:object_entity["start_idx"]+13] + "<obj>" + object_entity["word"] + "</obj>" + sentence[object_entity["end_idx"]+14:]
    return sentence


def get_label_list(labels, task):
    if task == 3:
        unique_labels = set()
        for label in labels:
            unique_labels = unique_labels | set(label)
        label_list = list(unique_labels)
        label_list.sort()
        return label_list
    else:
        return list(set(labels))


def str2bool(i):
    if isinstance(i, bool):
        return i
    if i.lower() in ('true', 't', '1'):
        return True
    elif i.lower() in ('false', 'f', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('It is not Boolean')

