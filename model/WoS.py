import os
import json
import pickle

import numpy as np
from tqdm import tqdm

import torch
from torch._C import device
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.autograd import Variable
from torch.optim import lr_scheduler
from torch.nn import CrossEntropyLoss
from torch.utils.data import Dataset

from transformers import AutoTokenizer, BertForPreTraining
from transformers.modeling_outputs import TokenClassifierOutput
from transformers.models.bert import BertPreTrainedModel, BertModel
from transformers.utils.dummy_pt_objects import load_tf_weights_in_transfo_xl


PAD_token = 1
SOS_token = 3
EOS_token = 2
UNK_token = 0 

# Custom Dataset class
class Dataset(Dataset):
    """Custom data.Dataset compatible with data.DataLoader."""
    def __init__(self, data_info, src_word2id, trg_word2id, sequicity, mem_word2id):
        """Reads source and target sequences from txt files."""
        self.ID = data_info['ID']
        self.turn_domain = data_info['turn_domain']
        self.turn_id = data_info['turn_id']
        self.dialog_history = data_info['dialog_history']
        self.turn_belief = data_info['turn_belief']
        self.gating_label = data_info['gating_label']
        self.turn_uttr = data_info['turn_uttr']
        self.generate_y = data_info["generate_y"]
        self.sequicity = sequicity
        self.num_total_seqs = len(self.dialog_history)
        self.src_word2id = src_word2id
        self.trg_word2id = trg_word2id
        self.mem_word2id = mem_word2id
    
    def __getitem__(self, index):
        """Returns one data pair (source and target)."""
        ID = self.ID[index]
        turn_id = self.turn_id[index]
        turn_belief = self.turn_belief[index]
        gating_label = self.gating_label[index]
        turn_uttr = self.turn_uttr[index]
        turn_domain = self.preprocess_domain(self.turn_domain[index])
        generate_y = self.generate_y[index]
        generate_y = self.preprocess_slot(generate_y, self.trg_word2id)
        context = self.dialog_history[index] 
        context = self.preprocess(context, self.src_word2id)
        context_plain = self.dialog_history[index]
        
        item_info = {
            "ID":ID, 
            "turn_id":turn_id, 
            "turn_belief":turn_belief, 
            "gating_label":gating_label, 
            "context":context, 
            "context_plain":context_plain, 
            "turn_uttr_plain":turn_uttr, 
            "turn_domain":turn_domain, 
            "generate_y":generate_y, 
            }
        return item_info

    def __len__(self):
        return self.num_total_seqs
    
    def preprocess(self, sequence, word2idx):
        """Converts words to ids."""
        story = [word2idx[word] if word in word2idx else UNK_token for word in sequence.split()]
        story = torch.Tensor(story)
        return story

    def preprocess_slot(self, sequence, word2idx):
        """Converts words to ids."""
        story = []
        for value in sequence:
            v = [word2idx[word] if word in word2idx else UNK_token for word in value.split()] + [EOS_token]
            story.append(v)
        # story = torch.Tensor(story)
        return story

    def preprocess_memory(self, sequence, word2idx):
        """Converts words to ids."""
        story = []
        for value in sequence:
            d, s, v = value
            s = s.replace("book","").strip()
            # separate each word in value to different memory slot
            for wi, vw in enumerate(v.split()):
                idx = [word2idx[word] if word in word2idx else UNK_token for word in [d, s, "t{}".format(wi), vw]]
                story.append(idx)
        story = torch.Tensor(story)
        return story

    def preprocess_domain(self, turn_domain):
        domains = {"숙소":0, "식당":1, "택시":2, "지하철":3, "관광":4}
        if isinstance(turn_domain, list):
            # If there's no turn_domain, return 5
            if not turn_domain:
                return 5
            for i in turn_domain:
                domain_list = domains[i]
            return domain_list
        return domains[turn_domain]

def collate_fn(data):
    def merge(sequences):
        '''
        merge from batch * sent_len to batch * max_len 
        '''
        lengths = [len(seq) for seq in sequences]
        max_len = 1 if max(lengths)==0 else max(lengths)
        padded_seqs = torch.ones(len(sequences), max_len).long()
        for i, seq in enumerate(sequences):
            end = lengths[i]
            padded_seqs[i, :end] = seq[:end]
        padded_seqs = padded_seqs.detach() #torch.tensor(padded_seqs)
        return padded_seqs, lengths

    def merge_multi_response(sequences):
        '''
        merge from batch * nb_slot * slot_len to batch * nb_slot * max_slot_len
        '''
        lengths = []
        for bsz_seq in sequences:
            length = [len(v) for v in bsz_seq]
            lengths.append(length)
        max_len = max([max(l) for l in lengths])
        padded_seqs = []
        for bsz_seq in sequences:
            pad_seq = []
            for v in bsz_seq:
                v = v + [PAD_token] * (max_len-len(v))
                pad_seq.append(v)
            padded_seqs.append(pad_seq)
        padded_seqs = torch.tensor(padded_seqs)
        lengths = torch.tensor(lengths)
        return padded_seqs, lengths

    def merge_memory(sequences):
        lengths = [len(seq) for seq in sequences]
        max_len = 1 if max(lengths)==0 else max(lengths) # avoid the empty belief state issue
        padded_seqs = torch.ones(len(sequences), max_len, 4).long()
        for i, seq in enumerate(sequences):
            end = lengths[i]
            if len(seq) != 0:
                padded_seqs[i,:end,:] = seq[:end]
        return padded_seqs, lengths
  
    # sort a list by sequence length (descending order) to use pack_padded_sequence
    data.sort(key=lambda x: len(x['context']), reverse=True) 
    item_info = {}
    for key in data[0].keys():
        item_info[key] = [d[key] for d in data]

    # merge sequences
    src_seqs, src_lengths = merge(item_info['context'])
    y_seqs, y_lengths = merge_multi_response(item_info["generate_y"])
    gating_label = torch.tensor(item_info["gating_label"])
    turn_domain = torch.tensor(item_info["turn_domain"])

    src_seqs = src_seqs.cuda()
    gating_label = gating_label.cuda()
    turn_domain = turn_domain.cuda()
    y_seqs = y_seqs.cuda()
    y_lengths = y_lengths.cuda()

    item_info["context"] = src_seqs
    item_info["context_len"] = src_lengths
    item_info["gating_label"] = gating_label
    item_info["turn_domain"] = turn_domain
    item_info["generate_y"] = y_seqs
    item_info["y_lengths"] = y_lengths

    return item_info

# Word Embedding used in TRADE
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

class TradeWos(nn.Module):
    def __init__(self, model_name, hidden_size, lang, model_save_path, lr, dropout, slots, gating_dict, max_sequence_length=512):
        super(TradeWos, self).__init__()
        self.model_name = model_name
        self.hidden_size = hidden_size
        self.lang = lang[0]
        self.mem_lang = lang[1]
        self.lr = lr
        self.dropout = dropout
        self.slots = slots[0]
        self.slot_temp = slots[1]
        self.gating_dict = gating_dict
        self.nb_gate = len(gating_dict)
        self.max_sequence_length = max_sequence_length
        self.loss_fcn = nn.CrossEntropyLoss()

        self.encoder = BertWos(self.model_name, self.lang.n_words, hidden_size, self.dropout, self.max_sequence_length)
        self.decoder = Generator(self.lang, self.encoder.embedding, self.lang.n_words, hidden_size, self.dropout, self.slots, self.nb_gate)

        if model_save_path and False:
            print(f"... Model saved in {model_save_path} is loaded ...")
            trained_encoder = torch.load(f"{model_save_path}enc.th")
            trained_decoder = None

        # Initialize optimizers and criterion
        self.optimizer = optim.AdamW(self.parameters(), lr=lr)
        self.scheduler = lr_scheduler.ReduceLROnPlateau(self.optimizer, mode="max", factor=0.5, patience=1, min_lr=0.0001, verbose=True)

        self.reset()
        self.encoder.cuda()
        self.decoder.cuda()

    def reset(self):
        self.loss, self.print_every, self.loss_ptr, self.loss_gate, self.loss_class = 0, 1, 0, 0, 0

    def print_loss(self):
        print_loss_avg = self.loss / self.print_every
        print_loss_ptr = self.loss_ptr / self.print_every
        print_loss_gate = self.loss_gate / self.print_every
        print_loss_class = self.loss_class / self.print_every
        self.print_every += 1
        print_string = f"Loss {print_loss_avg:.2f}, Loss Ptr {print_loss_ptr:.2f}, Loss gate {print_loss_gate:.2f}, Loss Class {print_loss_class:.2f}"
        return print_string
    
    def train_batch(self, batch_data, clip, slot_temp, reset=0):
        # All losses to zeros
        if reset:
            self.reset()

        self.optimizer.zero_grad()

        # Encode and Decode
        all_points_outputs, gates, words_point_output, words_class_out = self.encode_and_decode(batch_data, slot_temp)

        loss_ptr = self.masked_cross_entropy_for_value(
            all_points_outputs.transpose(0,1).contiguous(),
            batch_data["generate_y"].contiguous(),
            batch_data["y_lengths"]
        )
        loss_gate = self.cross_entropy(gates.transpose(0,1).contiguous().view(-1,gates.size(-1)), batch_data["gating_label"].contiguous().view(-1))

        loss = loss_ptr + loss_gate

        self.loss_grad = loss
        self.loss_ptr_to_bp = loss_ptr

        # Update parameters with optimizers
        self.loss += loss.data
        self.loss_ptr += loss_ptr.item()
        self.loss_gate += loss_gate.item()

    def optimize(self, clip):
        self.loss_grad.backward()
        clip_norm = torch.nn.utils.clip_grad_norm_(self.parameters(), clip)
        self.optimizer.step()

    def encode_and_decode(self, data, slot_temp):
        # If needed, put codes for masking

        # Encode dialog history
        #encoded_outputs, encoded_hidden = self.encoder(data['context_plain'], data['context_len'])
        encoded_outputs, encoded_hidden = self.encoder(data['context'].transpose(0,1), data['context_len'])

        # Get the words that can be copy from the memory
        batch_size = len(data['context_len'])
        self.copy_list = data['context_plain']
        max_res_len = data['generate_y'].size(2) if self.encoder.training else 10
        all_point_outputs, all_gate_outputs, words_point_out, words_class_out = self.decoder.forward(batch_size,\
            encoded_hidden, encoded_outputs, data['context_len'], data['context'], max_res_len, data['generate_y'],
            slot_temp)

        return all_point_outputs, all_gate_outputs, words_point_out, words_class_out

    def masked_cross_entropy_for_value(self, logits, target, mask):
        # logits_flat: (batch * max_len, num_classes)
        # -> logits_flat: (batch * slot_len * max_res_len, vocab_size)
        #logits_flat = logits.view(-1, logits.size(-1))
        # log_probs_flat: same as logits_flat
        #log_probs_flat = F.log_softmax(logits_flat, dim=1)
        # target_flat: (batch * max_len, 1)
        # -> target_flat: (batch * slot_len * max_res_len, 1 )
        #target_flat = target.view(-1, 1)
        # losses_flat: (batch * max_len, 1)
        # Find the word vector probability
        #losses_flat = -torch.gather(log_probs_flat, dim=1, index=target_flat)
        # losses: (batch, max_len)
        # losses: (batch, slot_len, max_res_len)
        #losses = losses_flat.view(target.size())
        # length: (batch, slot_len)
        # TODO: What is mask for?
        #mask = self.sequence_mask(sequence_length=length, max_len=target.size(1))
        #losses = losses * mask.float()
        #loss = losses.sum() / length.float().sum()        
        #return loss

        logits_flat = logits.view(-1, logits.size(-1)) ## -1 means infered from other dimentions
        log_probs_flat = torch.log(logits_flat)
        target_flat = target.view(-1, 1)
        losses_flat = -torch.gather(log_probs_flat, dim=1, index=target_flat)
        losses = losses_flat.view(*target.size()) # b * |s| * m
        loss = self.masking(losses, mask)
        return loss

    def masking(self, losses, mask):
        mask_ = []
        batch_size = mask.size(0)
        max_len = losses.size(2)
        for si in range(mask.size(1)):
            seq_range = torch.arange(0, max_len).long()
            seq_range_expand = seq_range.unsqueeze(0).expand(batch_size, max_len)
            if mask[:,si].is_cuda:
                seq_range_expand = seq_range_expand.cuda()
            seq_length_expand = mask[:, si].unsqueeze(1).expand_as(seq_range_expand)
            mask_.append( (seq_range_expand < seq_length_expand) )
        mask_ = torch.stack(mask_)
        mask_ = mask_.transpose(0, 1)
        if losses.is_cuda:
            mask_ = mask_.cuda()
        losses = losses * mask_.float()
        loss = losses.sum() / (mask_.sum().float())
        return loss

    def cross_entropy(self, logits, target):
        batch_size = logits.size(0)
        log_probs_flat = F.log_softmax(logits, dim=1)
        losses_flat = -torch.gather(log_probs_flat, dim=1, index=target.unsqueeze(1))
        loss = losses_flat.sum() / batch_size
        return loss

    def evaluate(self, dev, matric_best, slot_temp, early_stop=None):
        # Set to not-training mode to disable dropout
        self.encoder.train(False)
        self.decoder.train(False)  
        print("STARTING EVALUATION")
        all_prediction = {}
        inverse_unpoint_slot = dict([(v, k) for k, v in self.gating_dict.items()])
        pbar = tqdm(enumerate(dev),total=len(dev))
        #for j, data_dev in pbar: 
        for data_dev in tqdm(dev):
            # Encode and Decode
            batch_size = len(data_dev['context_len'])
            _, gates, words, class_words = self.encode_and_decode(data_dev, slot_temp)

            for bi in range(batch_size):
                if data_dev["ID"][bi] not in all_prediction.keys():
                    all_prediction[data_dev["ID"][bi]] = {}
                all_prediction[data_dev["ID"][bi]][data_dev["turn_id"][bi]] = {"turn_belief":data_dev["turn_belief"][bi]}
                predict_belief_bsz_ptr, predict_belief_bsz_class = [], []
                gate = torch.argmax(gates.transpose(0, 1)[bi], dim=1)

                # pointer-generator results
                for si, sg in enumerate(gate):
                    if sg==self.gating_dict["none"]:
                        continue
                        # TODO: yes or no 추가
                    elif sg==self.gating_dict["ptr"]:
                        pred = np.transpose(words[si])[bi]
                        st = []
                        for e in pred:
                            if e== 'EOS': break
                            else: st.append(e)
                        st = " ".join(st)
                        if st == "none":
                            continue
                        else:
                            predict_belief_bsz_ptr.append(slot_temp[si]+"-"+str(st))
                    else:
                        predict_belief_bsz_ptr.append(slot_temp[si]+"-"+inverse_unpoint_slot[sg.item()])


                all_prediction[data_dev["ID"][bi]][data_dev["turn_id"][bi]]["pred_bs_ptr"] = predict_belief_bsz_ptr

                """if set(data_dev["turn_belief"][bi]) != set(predict_belief_bsz_ptr):
                    print("True", set(data_dev["turn_belief"][bi]) )
                    print("Pred", set(predict_belief_bsz_ptr), "\n")"""

        joint_acc_score_ptr, F1_score_ptr, turn_acc_score_ptr = self.evaluate_metrics(all_prediction, "pred_bs_ptr", slot_temp)

        evaluation_metrics = {"Joint Acc":joint_acc_score_ptr, "Turn Acc":turn_acc_score_ptr, "Joint F1":F1_score_ptr}
        print(evaluation_metrics)

        # Set back to training mode
        self.encoder.train(True)
        self.decoder.train(True)

        return evaluation_metrics

    def evaluate_metrics(self, all_prediction, from_which, slot_temp):
        total, turn_acc, joint_acc, F1_pred, F1_count = 0, 0, 0, 0, 0
        for d, v in all_prediction.items():
            for t in range(len(v)):
                cv = v[t]
                if set(cv["turn_belief"]) == set(cv[from_which]):
                    joint_acc += 1
                total += 1

                # Compute prediction slot accuracy
                temp_acc = self.compute_acc(set(cv["turn_belief"]), set(cv[from_which]), slot_temp)
                turn_acc += temp_acc

                # Compute prediction joint F1 score
                temp_f1, temp_r, temp_p, count = self.compute_prf(set(cv["turn_belief"]), set(cv[from_which]))
                F1_pred += temp_f1
                F1_count += count

        joint_acc_score = joint_acc / float(total) if total!=0 else 0
        turn_acc_score = turn_acc / float(total) if total!=0 else 0
        F1_score = F1_pred / float(F1_count) if F1_count!=0 else 0
        return joint_acc_score, F1_score, turn_acc_score

    def compute_acc(self, gold, pred, slot_temp):
        miss_gold = 0
        miss_slot = []
        for g in gold:
            if g not in pred:
                miss_gold += 1
                miss_slot.append(g.rsplit("-", 1)[0])
        wrong_pred = 0
        for p in pred:
            if p not in gold and p.rsplit("-", 1)[0] not in miss_slot:
                wrong_pred += 1
        ACC_TOTAL = len(slot_temp)
        ACC = len(slot_temp) - miss_gold - wrong_pred
        ACC = ACC / float(ACC_TOTAL)
        return ACC

    def compute_prf(self, gold, pred):
        TP, FP, FN = 0, 0, 0
        if len(gold)!= 0:
            count = 1
            for g in gold:
                if g in pred:
                    TP += 1
                else:
                    FN += 1
            for p in pred:
                if p not in gold:
                    FP += 1
            precision = TP / float(TP+FP) if (TP+FP)!=0 else 0
            recall = TP / float(TP+FN) if (TP+FN)!=0 else 0
            F1 = 2 * precision * recall / float(precision + recall) if (precision+recall)!=0 else 0
        else:
            if len(pred)==0:
                precision, recall, F1, count = 1, 1, 1, 1
            else:
                precision, recall, F1, count = 0, 0, 0, 1
        return F1, recall, precision, count

class Generator(nn.Module):
    def __init__(self, lang, shared_emb, vocab_size, hidden_size, dropout, slots, nb_gate):
        super(Generator, self).__init__()
        self.vocab_size = vocab_size
        self.lang = lang
        self.embedding = shared_emb
        self.dropout_layer = nn.Dropout(dropout)
        self.gru = nn.GRU(hidden_size, hidden_size, dropout=dropout)
        self.nb_gate = nb_gate
        self.hidden_size = hidden_size
        self.W_ratio = nn.Linear(3*hidden_size, 1)
        self.softmax = nn.Softmax(dim=1)
        self.sigmoid = nn.Sigmoid()
        self.slots = slots

        self.W_gate = nn.Linear(hidden_size, nb_gate)

        # Create independent slot embeddings
        self.slot_w2i = {}
        for slot in self.slots:
            if slot.split('-')[0] not in self.slot_w2i.keys():
                self.slot_w2i[slot.split('-')[0]] = len(self.slot_w2i)
            if slot.split('-')[1] not in self.slot_w2i.keys():
                self.slot_w2i[slot.split('-')[1]] = len(self.slot_w2i)
        self.Slot_emb = nn.Embedding(len(self.slot_w2i), hidden_size)
        self.Slot_emb.weight.data.normal_(0, 0.1)

        self.Slot_emb.cuda()

    def forward(self, batch_size, encoded_hidden, encoded_outputs, encoded_lens, story, max_res_len, target_batches, slot_temp):
        all_points_outputs = torch.zeros(len(slot_temp), batch_size, max_res_len, self.vocab_size).cuda()
        all_gate_outputs = torch.zeros(len(slot_temp), batch_size, self.nb_gate).cuda()

        # Get the slot embedding
        slot_emb_dict = {}
        for i, slot in enumerate(slot_temp):
            # Domain embedding
            if slot.split('-')[0] in self.slot_w2i.keys():
                domain_w2idx = torch.Tensor([self.slot_w2i[slot.split('-')[0]]]).long().cuda()
                domain_emb = self.Slot_emb(domain_w2idx)
            # Slot embedding
            if slot.split('-')[1] in self.slot_w2i.keys():
                slot_w2idx = torch.Tensor([self.slot_w2i[slot.split('-')[1]]]).long().cuda()
                slot_emb = self.Slot_emb(slot_w2idx)

            # Combine two embeddings as one query
            combined_emb = domain_emb + slot_emb
            slot_emb_dict[slot] = combined_emb
            slot_emb_exp = combined_emb.expand_as(encoded_hidden)
            if i == 0:
                slot_emb_arr = slot_emb_exp.clone()
            else:
                slot_emb_arr = torch.cat((slot_emb_arr, slot_emb_exp), dim=0)
        
        
        # Compute pointer-generator output, decoding each (domain, slot) one-by-one
        words_point_out = []
        counter = 0
        for slot in slot_temp:
            hidden = encoded_hidden
            words = []
            slot_emb = slot_emb_dict[slot]
            decoder_input = self.dropout_layer(slot_emb).expand(batch_size, self.hidden_size)
            
            for wi in range(max_res_len):
                # Hidden states
                dec_state, hidden = self.gru(decoder_input.expand_as(hidden), hidden)
                # Vocaburary space
                p_vocab = self.attend_vocab(self.embedding.weight, hidden.squeeze(0))
                # History attention
                context_vec, logits, prob = self.attend(encoded_outputs, hidden.squeeze(0), encoded_lens)
                # Context vector
                if wi == 0:
                    all_gate_outputs[counter] = self.W_gate(context_vec)
                # Generate vector
                # TODO: Why dec_state? Isn't it hidden?
                p_gen_vec = torch.cat([dec_state.squeeze(0), context_vec, decoder_input], -1)
                vocab_pointer_switches = self.sigmoid(self.W_ratio(p_gen_vec))
                # Soft-gated pointer-generator copyping to combine a distribution over the vocaburary and
                # a distribution over the dialogue history into a single output distribution
                p_context_ptr = torch.zeros(p_vocab.size()).cuda()
                p_context_ptr.scatter_add_(1, story, prob)
                # Fianl vocab
                final_p_vocab = (1 - vocab_pointer_switches).expand_as(p_context_ptr) * p_context_ptr + \
                    vocab_pointer_switches.expand_as(p_context_ptr) * p_vocab
                pred_word = torch.argmax(final_p_vocab, dim=1)
                
                words.append([self.lang.index2word[w_idx.item()] for w_idx in pred_word])
                all_points_outputs[counter, :, wi, :] = final_p_vocab
                decoder_input = self.embedding(pred_word).cuda()

            counter += 1
            words_point_out.append(words)

        return all_points_outputs, all_gate_outputs, words_point_out, []

    def attend(self, seq, cond, lens):
        # Attend over the sequences 'seq' using the condition 'cond'
        scores_ = cond.unsqueeze(1).expand_as(seq).mul(seq).sum(2)
        max_len = max(lens)
        for i, l in enumerate(lens):
            if l < max_len:
                scores_.data[i, l:] = -np.inf
        scores = F.softmax(scores_, dim=1)
        context = scores.unsqueeze(2).expand_as(seq).mul(seq).sum(1)
        return context, scores_, scores

    def attend_vocab(self, seq, cond):
        scores_ = cond.matmul(seq.transpose(1,0))
        scores = F.softmax(scores_, dim=1)
        return scores
                


class BertWos(nn.Module):
    def __init__(self, model_name, vocab_size, hidden_size, dropout, max_sequence_length):
        super(BertWos, self).__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.dropout_layer = nn.Dropout(dropout)
        self.embedding = nn.Embedding(vocab_size, hidden_size, padding_idx=PAD_token)
        self.embedding.weight.data.normal_(0, 0.1)
        self.max_sequence_length = max_sequence_length
        self.bert = BertForPreTraining.from_pretrained(f"klue/{model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(f"klue/{model_name}")

        self.gru = nn.GRU(self.hidden_size, self.hidden_size, 1, dropout=self.dropout, bidirectional=True)

    def get_state(self, batch_size):
        # Get cell states and hidden states
        return Variable(torch.zeros(2, batch_size, self.hidden_size)).cuda()

    def forward(self, input_seqs, input_lengths):
        embedded = self.embedding(input_seqs)
        embedded = self.dropout_layer(embedded)
        hidden = self.get_state(input_seqs.size(1))
        if input_lengths:
            embedded = nn.utils.rnn.pack_padded_sequence(embedded, input_lengths, batch_first=False)
        outputs, hidden = self.gru(embedded, hidden)
        if input_lengths:
            outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs, batch_first=False)
        hidden = hidden[0] + hidden[1]
        outputs = outputs[:,:,:self.hidden_size] + outputs[:,:,:self.hidden_size]
        outputs = outputs.transpose(0,1)
        hidden = hidden.unsqueeze(0)
        return outputs, hidden
        """
        tokenized_input_seqs = self.tokenizer(input_seqs, padding=True, max_length=self.max_sequence_length, truncation=True, return_tensors='pt').to('cuda')
        outputs = self.bert(**tokenized_input_seqs, output_hidden_states=True)
        print(outputs)
        print(f"input_ids {tokenized_input_seqs.input_ids.size()}")
        print(f"prediction_logits {outputs.prediction_logits.size()}")
        print(f"hidden states {outputs.hidden_states[0].size()}")
        """

def preprocessing_WoS(dataset, batch_size, sequicity=0, is_train=True):
    # Load domain-slot pairs from ontology
    ontology_path = "data/wos/ontology.json"
    if not os.path.exists(ontology_path):
        make_ontology(dataset)
    ontology = json.load(open(ontology_path, "r"))
    D_S_PAIR = [k.replace(" ", "") for k in ontology.keys()]
    gating_dict = {"ptr": 0, "dontcare": 1, "none": 2, "yes": 3, "no": 4}

    # Vocaburary
    lang, mem_lang = Lang(), Lang()
    lang.index_words(D_S_PAIR, 'slot')
    mem_lang.index_words(D_S_PAIR, 'slot')
    lang_name = 'data/wos/lang-all.pkl'
    mem_lang_name = 'data/wos/mem-lang-all.pkl'

    if is_train:
        pair_train, train_max_len, slot_train = read_langs(dataset, gating_dict, D_S_PAIR, "train", lang, mem_lang, is_train)
        train_dataset = get_seq(pair_train, lang, mem_lang, sequicity)
        nb_train_vocab = lang.n_words
        
        pair_val, val_max_len, slot_val = read_langs(dataset, gating_dict, D_S_PAIR, "validation", lang, mem_lang, is_train)
        val_dataset = get_seq(pair_val, lang, mem_lang, sequicity)

        if os.path.exists(lang_name) and os.path.exists(mem_lang_name):
            print("... Loading saved lang files ...")
            with open(lang_name, 'rb') as f:
                lang = pickle.load(f)
            with open(mem_lang_name, 'rb') as f:
                mem_lang = pickle.load(f)
        else:
            print("... Dumping lang files ...")
            with open(lang_name, 'wb') as f:
                pickle.dump(lang, f)
            with open(mem_lang_name, 'wb') as f:
                pickle.dump(mem_lang, f)
        # If needed, put embedding save code here
    else:
        with open(lang_name, 'rb') as f:
            lang = pickle.load(f)
        with open(mem_lang_name, 'rb') as f:
            mem_lang = pickle.load(f)

        pair_train, train_max_len, slot_train, train, nb_train_vocab = [], 0, {}, [], 0
        pair_val, val_max_len, slot_val = read_langs(dataset, gating_dict, D_S_PAIR, "validation", lang, mem_lang, is_train)
        val_dataset = get_seq(pair_val, lang, mem_lang, sequicity)

    max_word = max(train_max_len, val_max_len) + 1

    print(f"... Read {len(pair_train)} pairs in train ...")
    print(f"... Read {len(pair_val)} pairs in validation ...")
    print(f"... Vocab size is {lang.n_words} ...")
    print(f"... Vocab size of training dataset is {nb_train_vocab} ...")
    print(f"... Vocab size of states is {mem_lang.n_words} ...")
    print(f"... Max word length of dialog words is {max_word} ...")

    SLOTS_LIST = [D_S_PAIR, slot_train, slot_val]
    print(f"... Train slot length is {len(SLOTS_LIST[1])} & Validation slot length is {len(SLOTS_LIST[2])} ...")
    LANG = [lang, mem_lang]

    return train_dataset, val_dataset, LANG, SLOTS_LIST, gating_dict

def get_seq(pairs, lang, mem_lang, sequicity):
    data_info = {}
    data_keys = pairs[0].keys()
    for k in data_keys:
        data_info[k] = []

    for pair in pairs:
        for k in data_keys:
            data_info[k].append(pair[k])

    dataset = Dataset(data_info, lang.word2index, lang.word2index, sequicity, mem_lang.word2index)
    
    return dataset

def read_langs(dataset, gating_dict, D_S_PAIR, mode, lang, mem_lang, is_train, max_line=None):
    data = []
    max_resp_len, max_value_len = 0, 0
    domain_counter = {}

    # Create Word Embedding First
    if mode == "train" and is_train:
        for dial in dataset[mode]["dialogue"]:
            for ti, turn in enumerate(dial):
                lang.index_words(turn["text"], "utter")

    cnt_line = 1
    for dial_dict in dataset[mode]:
        dialogue_history = ""
        # Filtering and counting domains
        for domain in dial_dict["domains"]:
            if domain not in domain_counter.keys():
                domain_counter[domain] = 0
            domain_counter[domain] += 1

        # Reading data
        turn_state_all = []
        dialogue_turn_length = len(dial_dict["dialogue"]) // 2
        for ti in range(dialogue_turn_length):
            turn_id = ti * 2 if ti*2 < dialogue_turn_length else ti*2-1
            turn = dial_dict["dialogue"][turn_id]
            turn_prev = dial_dict["dialogue"][turn_id-1]

            if turn["state"]:
                if turn_id == 0:
                    turn_state_all = turn["state"]
                    turn_domain = turn_state_all[0].split("-")[0]
                else:
                    turn_state = [s for s in turn["state"] if s not in turn_state_all]
                    turn_state_all += turn_state
                    turn_domain = [i.split('-')[0] for i in turn_state]
                    turn_domain = list(set(turn_domain))

            if turn_id == 0 or turn_id % 2 == 1:
                turn_utter = " ; " + turn["text"] if turn["role"] == "user" else turn["text"] + " ; "
            else:
                turn_utter = turn["text"] + " ; " + turn_prev["text"] if turn["role"] == "sys" else turn_prev["text"] + " ; " + turn["text"]
            turn_utter_strip = turn_utter.strip()
            dialogue_history += (turn_utter + " ; ")
            source_text = dialogue_history.strip()
            slot_temp = D_S_PAIR
            turn_state_list = turn["state"] if turn["role"] == "user" else turn_prev["state"]
            turn_state_dict = dict([(l.rsplit('-', 1)[0].replace(" ", ""), l.rsplit('-', 1)[1]) for l in turn_state_list])
            
            if is_train:
                mem_lang.index_words(turn_state_dict, 'belief')

            generate_y, gating_label = [], []
            for slot in slot_temp:
                if slot in turn_state_dict.keys():
                    generate_y.append(turn_state_dict[slot])

                    if turn_state_dict[slot] == "dontcare":
                        gating_label.append(gating_dict["dontcare"])
                    elif turn_state_dict[slot] == "none":
                        gating_label.append(gating_dict["none"])
                    elif turn_state_dict[slot] == "yes":
                        gating_label.append(gating_dict["yes"])
                    elif turn_state_dict[slot] == "no":
                        gating_label.append(gating_dict["no"])
                    else:
                        gating_label.append(gating_dict["ptr"])

                    if max_value_len < len(turn_state_dict[slot]):
                        max_value_len = len(turn_state_dict[slot])

                else:
                    generate_y.append("none")
                    gating_label.append(gating_dict["none"])

            data_detail = {
                "ID": dial_dict["guid"],
                "domains": dial_dict["domains"],
                "turn_domain": turn_domain,
                "turn_id": ti,
                "dialog_history": source_text,
                "turn_belief": turn_state_list,
                "gating_label": gating_label,
                "turn_uttr": turn_utter_strip,
                "generate_y": generate_y,
            }
            data.append(data_detail)

            if max_resp_len < len(source_text.split()):
                max_resp_len = len(source_text.split())

        cnt_line += 1
        if (max_line and cnt_line >= max_line): 
            break

    if "t{}".format(max_value_len-1) not in mem_lang.word2index.keys() and is_train:
        for time_i in range(max_value_len):
            mem_lang.index_words("t{}".format(time_i), 'utter')

    return data, max_resp_len, slot_temp

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

