import os
import json
from collections import OrderedDict
import numpy as np
from numpy.random import randint, beta

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import MSELoss, CrossEntropyLoss

from transformers.configuration_utils import PretrainedConfig
from transformers.modeling_utils import PreTrainedModel
from transformers.modeling_outputs import BaseModelOutputWithPoolingAndCrossAttentions ,SequenceClassifierOutput
from transformers.models.bert import BertConfig, BertPreTrainedModel, BertModel, load_tf_weights_in_bert
from transformers.models.roberta import RobertaConfig, RobertaModel
from transformers.models.roberta.modeling_roberta import RobertaPreTrainedModel, RobertaClassificationHead



LOSS_JSON = OrderedDict()
WEIGHT = OrderedDict()


BERT_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "bert-base-uncased": "https://huggingface.co/bert-base-uncased/resolve/main/config.json",
    "bert-large-uncased": "https://huggingface.co/bert-large-uncased/resolve/main/config.json",
    "bert-base-cased": "https://huggingface.co/bert-base-cased/resolve/main/config.json",
    "bert-large-cased": "https://huggingface.co/bert-large-cased/resolve/main/config.json",
    "bert-base-multilingual-uncased": "https://huggingface.co/bert-base-multilingual-uncased/resolve/main/config.json",
    "bert-base-multilingual-cased": "https://huggingface.co/bert-base-multilingual-cased/resolve/main/config.json",
    "bert-base-chinese": "https://huggingface.co/bert-base-chinese/resolve/main/config.json",
    "bert-base-german-cased": "https://huggingface.co/bert-base-german-cased/resolve/main/config.json",
    "bert-large-uncased-whole-word-masking": "https://huggingface.co/bert-large-uncased-whole-word-masking/resolve/main/config.json",
    "bert-large-cased-whole-word-masking": "https://huggingface.co/bert-large-cased-whole-word-masking/resolve/main/config.json",
    "bert-large-uncased-whole-word-masking-finetuned-squad": "https://huggingface.co/bert-large-uncased-whole-word-masking-finetuned-squad/resolve/main/config.json",
    "bert-large-cased-whole-word-masking-finetuned-squad": "https://huggingface.co/bert-large-cased-whole-word-masking-finetuned-squad/resolve/main/config.json",
    "bert-base-cased-finetuned-mrpc": "https://huggingface.co/bert-base-cased-finetuned-mrpc/resolve/main/config.json",
    "bert-base-german-dbmdz-cased": "https://huggingface.co/bert-base-german-dbmdz-cased/resolve/main/config.json",
    "bert-base-german-dbmdz-uncased": "https://huggingface.co/bert-base-german-dbmdz-uncased/resolve/main/config.json",
    "cl-tohoku/bert-base-japanese": "https://huggingface.co/cl-tohoku/bert-base-japanese/resolve/main/config.json",
    "cl-tohoku/bert-base-japanese-whole-word-masking": "https://huggingface.co/cl-tohoku/bert-base-japanese-whole-word-masking/resolve/main/config.json",
    "cl-tohoku/bert-base-japanese-char": "https://huggingface.co/cl-tohoku/bert-base-japanese-char/resolve/main/config.json",
    "cl-tohoku/bert-base-japanese-char-whole-word-masking": "https://huggingface.co/cl-tohoku/bert-base-japanese-char-whole-word-masking/resolve/main/config.json",
    "TurkuNLP/bert-base-finnish-cased-v1": "https://huggingface.co/TurkuNLP/bert-base-finnish-cased-v1/resolve/main/config.json",
    "TurkuNLP/bert-base-finnish-uncased-v1": "https://huggingface.co/TurkuNLP/bert-base-finnish-uncased-v1/resolve/main/config.json",
    "wietsedv/bert-base-dutch-cased": "https://huggingface.co/wietsedv/bert-base-dutch-cased/resolve/main/config.json",
    # See all BERT models at https://huggingface.co/models?filter=bert
}

ROBERTA_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "roberta-base",
    "roberta-large",
    "roberta-large-mnli",
    "distilroberta-base",
    "roberta-base-openai-detector",
    "roberta-large-openai-detector",
    # See all RoBERTa models at https://huggingface.co/models?filter=roberta
]


class MixupBertForSequenceClassification(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        self.init_weights()

    def forward(
        self,
        warm_up,
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

        pooled_output = outputs[1]

        # Mixup Process
        pooled_output_hat, labels_a, labels_b = mixup_data(pooled_output, labels)

        pooled_output = self.dropout(pooled_output)
        pooled_output_hat = self.dropout(pooled_output_hat)
        logits = self.classifier(pooled_output)
        logits_hat = self.classifier(pooled_output_hat)
        
        loss = None
        loss_mixup = None
        if labels is not None:
            if self.num_labels == 1:
                loss_fct = MSELoss()
                if warm_up:
                    #  We are doing regression
                    loss = loss_fct(logits.view(-1), labels.view(-1))
                else:
                    #loss_mixup = loss_fct(logits_hat.view(-1), labels_hat.view(-1))
                    loss_mixup = mixup_criterion(loss_fct, logits_hat.view(-1), labels_a.view(-1), labels_b.view(-1))
            else:
                loss_fct = CrossEntropyLoss()
                if warm_up:
                    loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
                else:
                    #loss_mixup = loss_fct(logits_hat.view(-1, self.num_labels), labels_hat.view(-1))
                    loss_mixup = mixup_criterion(loss_fct, logits_hat.view(-1, self.num_labels), labels_a.view(-1), labels_b.view(-1))

        if not return_dict:
            output = (logits,) + outputs[2:]
            output_mixup = (logits_hat,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output, ((loss_mixup,) + output_mixup) if loss_mixup is not None else output_mixup
        
        if warm_up:
            return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

        return SequenceClassifierOutput(
            loss=loss_mixup,
            logits=logits_hat,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions, )


class LogitMixBertForSequenceClassification(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        """self.layer_1 = nn.Linear(config.hidden_size, int(config.hidden_size/2))
        self.layer_2 = nn.Linear(int(config.hidden_size/2), int(config.hidden_size/4))
        self.classifier = nn.Linear(int(config.hidden_size/4), config.num_labels)"""

        self.init_weights()

    def forward(
        self,
        warm_up,
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

        """ LogitMix Process """
        #weight = [WEIGHT['mixed_loss'], WEIGHT['ce_loss'], WEIGHT['similarity_loss']]
        weight = [0.5, 0.5, 0.5]

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

        pooled_output = outputs[1]
        mixed_outputs, prob, index = logitmix_data(pooled_output, alpha=3, dist='beta')

        pooled_output = self.dropout(pooled_output)
        pooled_output_hat = self.dropout(mixed_outputs)
        logits = self.classifier(pooled_output)
        logits_hat = self.classifier(pooled_output_hat)

        """layer_1_output = self.layer_1(pooled_output)
        layer_1_output_hat = self.layer_1(pooled_output_hat)
        layer_2_output = self.layer_2(layer_1_output)
        layer_2_output_hat = self.layer_2(layer_1_output_hat)

        logits = self.classifier(layer_2_output)
        logits_hat = self.classifier(layer_2_output_hat)"""
      
        loss = None
        loss_mixup = None
        if labels is not None:
            if warm_up:
                if self.num_labels == 1:
                    #  We are doing regression
                    loss_fct = MSELoss()
                    loss = loss_fct(logits.view(-1), labels.view(-1))
                else:
                    loss_fct = CrossEntropyLoss()
                    loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
                loss_log(0, loss, 0)
            else:
                # LogitMix Process
                mixed_loss, org_loss, similarity = logitmix_criterion(logits, logits_hat, logits, logits_hat, labels, prob, index, self.num_labels)
                #print(f"mixed loss {weight[0]*mixed_loss} / org_loss {weight[1]*org_loss} / similarity {weight[2]*similarity}")
                loss_log(weight[0]*mixed_loss, weight[1]*org_loss, weight[2]*similarity)
                loss_mixup = weight[0] * mixed_loss + weight[1] * org_loss + weight[2] * similarity

        if not return_dict:
            output = (logits,) + outputs[2:]
            output_mixup = (logits_hat,) + mixed_outputs[2:]
            return ((loss,) + output) if loss is not None else output, ((loss_mixup,) + output_mixup) if loss_mixup is not None else output_mixup

        if warm_up:
            return SequenceClassifierOutput(
                loss=loss,
                logits=logits,
                hidden_states=outputs.hidden_states,
                attentions=outputs.attentions,
            )
        
        return SequenceClassifierOutput(
            loss=loss_mixup,
            logits=logits_hat,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions, )


class LogitMixRobertaForSequenceClassification(RobertaPreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.roberta = RobertaModel(config, add_pooling_layer=False)
        self.classifier = RobertaClassificationHead(config)

        self.init_weights()

    def forward(
        self,
        warm_up,
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

        """ LogitMix Process """
        weight = [WEIGHT['mixed_loss'], WEIGHT['ce_loss'], WEIGHT['similarity_loss']]

        outputs = self.roberta(
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
        mixed_outputs, prob, index = logitmix_data(sequence_output, alpha=3, dist='beta')

        logits = self.classifier(sequence_output)
        logits_hat = self.classifier(mixed_outputs)

        loss = None
        loss_mixup = None
        if labels is not None:
            if warm_up:
                if self.num_labels == 1:
                    #  We are doing regression
                    loss_fct = MSELoss()
                    loss = loss_fct(logits.view(-1), labels.view(-1))
                else:
                    loss_fct = CrossEntropyLoss()
                    loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
                loss_log(0, loss, 0)
            else:
                # LogitMix Process
                mixed_loss, org_loss, similarity = logitmix_criterion(logits, logits_hat, logits, logits_hat, labels, prob, index, self.num_labels)
                #print(f"mixed loss {weight[0]*mixed_loss} / org_loss {weight[1]*org_loss} / similarity {weight[2]*similarity}")
                loss_log(weight[0]*mixed_loss, weight[1]*org_loss, weight[2]*similarity)
                loss_mixup = weight[0] * mixed_loss + weight[1] * org_loss + weight[2] * similarity

        if not return_dict:
            output = (logits,) + outputs[2:]
            output_mixup = (logits_hat,) + mixed_outputs[2:]
            return ((loss,) + output) if loss is not None else output, ((loss_mixup,) + output_mixup) if loss_mixup is not None else output_mixup

        if warm_up:
            return SequenceClassifierOutput(
                loss=loss,
                logits=logits,
                hidden_states=outputs.hidden_states,
                attentions=outputs.attentions,
            )
        
        return SequenceClassifierOutput(
            loss=loss_mixup,
            logits=logits_hat,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions, )



def loss_log(mixed_loss, org_loss, similarity):
    ce_loss = org_loss.detach().cpu().numpy().tolist()
    if torch.is_tensor(mixed_loss):
        mixed_loss_ = mixed_loss.detach().cpu().numpy().tolist()
    else:
        mixed_loss_ = 0.0
    if torch.is_tensor(similarity):
        similarity_loss = similarity.detach().cpu().numpy().tolist()
    else:
        similarity_loss = 0.0

    dic = {'ce_loss': ce_loss, 'mixed_loss': mixed_loss_, 'similarity_loss': similarity_loss}
    LOSS_JSON[len(LOSS_JSON.keys())] = dic


def mixup_process(in_x, in_t, label_x, label_t, lambda_=0.5):
    lambda_ = beta(1.0, 1.0)
    x_hat = lambda_ * in_x + (1 - lambda_) * in_t
    y_hat = lambda_ * label_x + (1 - lambda_) * label_t
    return x_hat, y_hat


""" From mixup: beyond empirical risk minimization """
def mixup_data(x, y, lam=0.5, use_cuda=True):
    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b


def mixup_criterion(criterion, pred, y_a, y_b, lam=0.5):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


""" From LogitMix github ./utils.py """
def logitmix_data(x, alpha=.5, dist='beta', use_cuda=True):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if 'beta' in dist:
        lam = np.random.beta(alpha, alpha)
    elif 'normal' in dist:
        lam = np.clip(np.random.normal(0.5, alpha), 0, 1.)
    else:
        raise NotImplementedError
    lam = round(lam, 6)
    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda(x.device)
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    return mixed_x, lam, index


def logitmix_criterion(org_pred, mixed_pred, org_mid, mixed_mid, target, prob, index, num_labels, is_LS=False):
    batch_size = target.size(0)
    if is_LS:
        target = label_smoothing(org_pred, target)
    # labels
    y_a, y_b = target, target[index]
    if num_labels == 1:
        loss_function = MSELoss()
        mixed_loss = prob * loss_function(mixed_pred.view(-1), y_a.view(-1)) + (1-prob) * loss_function(mixed_pred.view(-1), y_b.view(-1))
        org_loss = loss_function(org_pred.view(-1), y_a.view(-1))
    else:
        # mixup loss
        mixed_loss = prob * cross_entropy(mixed_pred, y_a, is_LS=is_LS) + \
                    (1 - prob) * cross_entropy(mixed_pred, y_b, is_LS=is_LS)
        # cross entropy loss
        org_loss = cross_entropy(org_pred, y_a, is_LS=is_LS)
    # similarity loss
    similarity =  similarity_loss(org_mid, mixed_mid, index, prob)
    return mixed_loss, org_loss, similarity


def similarity_loss(original, mixed, index, prob):
    """ similartiy losses
        by using mse, kl_div...
    """
    mixed_org = prob * original + (1 - prob) * original[index]
    loss = sym_mse_loss(mixed, mixed_org)
    return loss


def sym_mse_loss(input_logits, target_logits):
    """ mse_loss for update input and target simultaneously
    """
    return torch.mean(torch.sum((input_logits - target_logits)**2, dim=1) / input_logits.size(1))


def cross_entropy(input, target, is_LS=False):
    """ Cross entropy for one-hot labels
    """
    if is_LS:
        target = label_smoothing(input, target)
        return -torch.mean(torch.sum(target * F.log_softmax(input, dim=1), dim=1))
    else:
        return F.cross_entropy(input, target)


def label_smoothing(pred, target, alpha=0.1):
    target_ = target.contiguous()
    one_hot = torch.zeros_like(pred).scatter(1, target_.view(-1, 1), 1)
    n_class = pred.size(1)
    smoothed_one_hot = one_hot * (1-alpha) + (1-one_hot) * alpha / (n_class - 1)
    return smoothed_one_hot
