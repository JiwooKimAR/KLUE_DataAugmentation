import os
import json
from typing import Dict

import torch
import torch.nn as nn
from torch import optim
from torch.optim import lr_scheduler
from torch.nn import CrossEntropyLoss

from transformers.modeling_outputs import TokenClassifierOutput
from transformers.models.bert import BertPreTrainedModel, BertModel, BertForPreTraining


PAD_token = 1
SOS_token = 3
EOS_token = 2
UNK_token = 0 


class BertWoS(nn.Module):
    def __init__(self, bert_model_path, hidden_size, lr, dropout, slots, gating_dict, nb_train_vocab=0):
        super(BertWoS, self).__init__()
        self.bert_model_path = bert_model_path
        self.hidden_size = hidden_size
        self.lr = lr
        self.dropout = dropout
        self.slots = slots
        self.gating_dict = gating_dict
        self.nb_train_vocab = nb_train_vocab
        self.loss_fcn = nn.CrossEntropyLoss()

        self.encoder = BertForPreTraining.from_pretrained(f"klue/{self.bert_model_path}")
        self.decoder = None

        self.encoder.cuda()
        self.decoder.cuda()

        self.optimizer = optim.AdamW(self.parameters(), lr=lr)
        self.scheduler = lr_scheduler.ReduceLROnPlateau(self.optimizer, mode="max", factor=0.5, patience=1, min_lr=0.0001, verbose=True)
    
    def train(self):
        self.optimizer.zero_grad()

        # Encode and Decode

    def optimize(self):
        a = 1

    def encode_and_decode(self, data, slot_temp):
        a = 1

    def evaluate(self):
        a = 1

    def evaluate_metrics(self):
        a = 1

    def compute_acc(self):
        a = 1

    def compute_prf(self):
        a = 1


class Bert(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.bert = BertModel(config, add_pooling_layer=True)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]
        pooled_output = outputs[1]

        sequence_output = self.dropout(sequence_output)

        print(pooled_output.shape)
        print(sequence_output.shape)
        exit()

        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            # Only keep active parts of the loss
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_logits = logits.view(-1, self.num_labels)
                active_labels = torch.where(
                    active_loss, labels.view(-1), torch.tensor(loss_fct.ignore_index).type_as(labels)
                )
                loss = loss_fct(active_logits, active_labels)
            else:
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


def preprocessing_WoS(dataset):
    # Load domain-slot pairs from ontology
    ontology_path = "data/wos/ontology.json"
    if not os.path.exists(ontology_path):
        make_ontology(dataset)
    ontology = json.load(open(ontology_path, "r"))
    D_S_PAIR = [k.replace(" ", "") for k in ontology.keys()]
    gating_dict = {"ptr": 0, "dontcare": 1, "none": 2, "yes": 3, "no": 4}

    # Vocaburary
    lang, mem_lang = Lang(), Lang()
    lang.index_word(D_S_PAIR, 'slot')
    mem_lang.index_word(D_S_PAIR, 'slot')


def make_ontology(dataset):
    data = dict()
    for k in ["train", "validation"]:
        for i in dataset[k]["dialogue"]:
            for j in i[0]["state"]:
                d_s_pair, obj = j.rsplit('-', 1)
                if not d_s_pair in data.keys():
                    data.update({d_s_pair: set()})
                else:
                    data[d_s_pair].add(obj)
    for i in data.keys():
        data[i] = list(data[i])
    with open("data/wos/ontology.json", "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


class Lang:
    def __init__(self):
        self.word2index = {}
        self.index2word = {PAD_token: "PAD", SOS_token: "SOS", EOS_token: "EOS", UNK_token: 'UNK'}
        self.n_words = len(self.index2word) # Count default tokens
        self.word2index = dict([(v, k) for k, v in self.index2word.items()])
      
    def index_words(self, sent, type):
        if type == 'utter':
            for word in sent.split(" "):
                self.index_word(word)
        elif type == 'slot':
            for slot in sent:
                d, s = slot.split("-")
                self.index_word(d)
                for ss in s.split(" "):
                    self.index_word(ss)
        elif type == 'belief':
            for slot, value in sent.items():
                d, s = slot.split("-")
                self.index_word(d)
                for ss in s.split(" "):
                    self.index_word(ss)
                for v in value.split(" "):
                    self.index_word(v)

    def index_word(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.index2word[self.n_words] = word
            self.n_words += 1