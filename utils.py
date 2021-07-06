import random
import argparse
import numpy as np
import torch

import datasets


NER_CLASSES = np.array(["B-PS", "I-PS", "B-LC", "I-LC", "B-OG", "I-OG", "B-DT", "I-DT", "B-TI", "I-TI", "B-QT", "I-QT", "O"])


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

