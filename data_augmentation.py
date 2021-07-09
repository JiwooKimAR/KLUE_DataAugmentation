import os
import pathlib
import json
import re
import random
import text2text as t2t

from nltk.corpus import wordnet
from datasets import Dataset

# Code from EDA
def get_only_chars(line):

    clean_line = ""

    line = line.replace("’", "")
    line = line.replace("'", "")
    line = line.replace("-", " ") #replace hyphens with spaces
    line = line.replace("\t", " ")
    line = line.replace("\n", " ")
    line = line.lower()

    for char in line:
        clean_line += char

    clean_line = re.sub(' +',' ',clean_line) #delete extra spaces
    if clean_line[0] == ' ':
        clean_line = clean_line[1:]
    return clean_line

def random_deletion(words, p):
	#obviously, if there's only one word, don't delete it
	if len(words) == 1:
		return words

	#randomly delete words with probability p
	new_words = []
	for word in words:
		r = random.uniform(0, 1)
		if r > p:
			new_words.append(word)

	#if you end up deleting all words, just return a random word
	if len(new_words) == 0:
		rand_int = random.randint(0, len(words)-1)
		return [words[rand_int]]

	return new_words

def RD(sentence, alpha_rd, n_aug):
    sentence = get_only_chars(sentence)
    words = sentence.split(' ')
    words = [word for word in words if word != '']
    num_words = len(words)
    
    augmented_sentences = []
    
    for _ in range(n_aug):
        a_words = random_deletion(words, alpha_rd)
        augmented_sentences.append(' '.join(a_words))
        
    augmented_sentences = [get_only_chars(sentence) for sentence in augmented_sentences]
    random.shuffle(augmented_sentences)
    
    augmented_sentences.append(sentence)
    
    return augmented_sentences

def swap_word(new_words):
    random_idx_1 = random.randint(0, len(new_words)-1)
    random_idx_2 = random_idx_1
    counter = 0
    while random_idx_2 == random_idx_1:
        random_idx_2 = random.randint(0, len(new_words)-1)
        counter += 1
        if counter > 3:
            return new_words
    new_words[random_idx_1], new_words[random_idx_2] = new_words[random_idx_2], new_words[random_idx_1]
    return new_words

def random_swap(words, n):
    new_words = words.copy()
    for _ in range(n):
        new_words = swap_word(new_words)
    return new_words

def RS(sentence, alpha_rs, n_aug):

    sentence = get_only_chars(sentence)
    words = sentence.split(' ')
    num_words = len(words)

    augmented_sentences = []
    n_rs = max(1, int(alpha_rs*num_words))

    for _ in range(n_aug):
        a_words = random_swap(words, n_rs)
        augmented_sentences.append(' '.join(a_words))

    augmented_sentences = [get_only_chars(sentence) for sentence in augmented_sentences]
    random.shuffle(augmented_sentences)

    augmented_sentences.append(sentence)

    return augmented_sentences


class DataAugmentationMethod:
    def __init__(
        self, dataset, dataset_name, sentence1=None, sentence2=None, bt=True, eda=[True, True, True, True],
         mixup=[True, True], alpha=0.3, eda_num=9,
        ):
        self.flag_bt = bt
        self.flag_sr = eda[0]
        self.flag_ri = eda[1]
        self.flag_rs = eda[2]
        self.flag_rd = eda[3]
        self.flag_lm = mixup[0]
        self.flag_mog = mixup[1]
        
        self.dataset_name = dataset_name
        self.dataset = dataset
        self.sentence1 = sentence1
        self.sentence2 = sentence2

        self.output_dir = f"data/{dataset_name}"

        # Augmentation configuration
        self.tgt_lang = ["en", "de"]
        self.bt_num = 1

        self.alpha = alpha # [0.05, 0.1, 0.2, 0.3, 0.4 ,0.5]
        self.eda_num = eda_num

        print("*** Direct Augmentation Class Initialize ***")

    def do_direct_aug(self):
        if self.back_translation():
            print("## Back Translation Done ##")
        if self.easy_data_augmentation():
            print("## Easy Data Augmentation Done ##")
        print("## Direct Augmentation is Done ##")
        
    def do_mixup_aug(self): # 생각을 좀 더 해보삼
        if self.flag_lm:
            return "LogitMix"
        elif self.flag_mog:
            return "MixOnGLUE"

    def is_exists(self, outdir):
        if os.path.isdir(outdir):
            if os.path.isfile(f"{outdir}train.json") and os.path.isfile(f"{outdir}validation.json"):
                return True
        return False


    def back_translation(self):
        if self.flag_bt == False:
            return False
        
        print("## Back Translation Start ##")
        outdir = self.output_dir + "_back_translation/"
        if self.is_exists(outdir):
            return True

        pathlib.Path(outdir).mkdir(exist_ok=True, parents=True)
        out_train = outdir + "train.json"
        out_validation = outdir + "validation.json"

        out_train_json = []
        out_validation_json = []
        # Back-Translation
        for k in range(len(self.dataset["train"][self.sentence1])):
            org = self.dataset["train"][k]
            if self.sentence2 == None:
                for i in self.tgt_lang:
                    for j in range(self.bt_num):
                        translated = t2t.Handler([self.dataset["train"][self.sentence1][k]], src_lang='ko').translate(tgt_lang=i)
                        out_line1 = t2t.Handler(translated, src_lang=i).translate(tgt_lang='ko')
                        org[self.sentence1] = out_line1
                        out_train_json.append(org.copy())
            else:
                for i in self.tgt_lang:
                    for j in range(self.bt_num):
                        translated = t2t.Handler([self.dataset["train"][self.sentence1][k]], src_lang='ko').translate(tgt_lang=i)
                        out_line1 = t2t.Handler(translated, src_lang=i).translate(tgt_lang='ko')
                        translated = t2t.Handler([self.dataset["train"][self.sentence2][k]], src_lang='ko').translate(tgt_lang=i)
                        out_line2 = t2t.Handler(translated, src_lang=i).translate(tgt_lang='ko')
                        org[self.sentence1] = out_line1
                        org[self.sentence2] = out_line2
                        out_train_json.append(org.copy())
            
        with open(out_train, 'w') as f:
            json.dump(out_train_json, f, indent=2, ensure_ascii=False)

        del out_train_json

        for k in range(len(self.dataset["validation"][self.sentence1])):
            org = self.dataset["validation"][k]
            if self.sentence2 == None:
                for i in self.tgt_lang:
                    for j in range(self.bt_num):
                        translated = t2t.Handler([self.dataset["validation"][self.sentence1][k]], src_lang='ko').translate(tgt_lang=i)
                        out_line1 = t2t.Handler(translated, src_lang=i).translate(tgt_lang='ko')
                        org[self.sentence1] = out_line1
                        out_validation_json.append(org.copy())
            else:
                for i in self.tgt_lang:
                    for j in range(self.bt_num):
                        translated = t2t.Handler([self.dataset["validation"][self.sentence1][k]], src_lang='ko').translate(tgt_lang=i)
                        out_line1 = t2t.Handler(translated, src_lang=i).translate(tgt_lang='ko')
                        translated = t2t.Handler([self.dataset["validation"][self.sentence2][k]], src_lang='ko').translate(tgt_lang=i)
                        out_line2 = t2t.Handler(translated, src_lang=i).translate(tgt_lang='ko')
                        org[self.sentence1] = out_line1
                        org[self.sentence2] = out_line2
                        out_validation_json.append(org.copy())

        with open(out_validation, 'w') as f:
            json.dump(out_validation_json, f, indent=2, ensure_ascii=False)

        del out_validation_json

        return True

    # Code from EDA
    def synonym_replacement(self):
        a = 1

    def random_insertion(self):
        a = 1

    def random_swap(self):
        outdir = self.output_dir + "_random_swap/"
        if self.is_exists(outdir):
            return True

        pathlib.Path(outdir).mkdir(exist_ok=True, parents=True)
        out_train = outdir + "train.json"
        out_validation = outdir + "validation.json"

        out_train_json = []
        out_validation_json = []

        # Random Swap: 어미 같은거 생각 안 함
        for k in range(len(self.dataset["train"][self.sentence1])):
            org = self.dataset["train"][k]
            if self.sentence2 == None:
                aug_sentences = RS(org[self.sentence1], self.alpha, self.eda_num)
                for s in aug_sentences:
                    org[self.sentence1] = s
                    out_train_json.append(org.copy())
            else:
                aug_sentences = RS(org[self.sentence1], self.alpha, self.eda_num)
                for s in aug_sentences:
                    org[self.sentence1] = s
                    out_train_json.append(org.copy())
                aug_sentences = RS(org[self.sentence2], self.alpha, self.eda_num)
                for s in aug_sentences:
                    org[self.sentence2] = s
                    out_train_json.append(org.copy())
            
        with open(out_train, 'w') as f:
            json.dump(out_train_json, f, indent=2, ensure_ascii=False)

        del out_train_json

        for k in range(len(self.dataset["validation"][self.sentence1])):
            org = self.dataset["validation"][k]
            if self.sentence2 == None:
                for j in range(self.bt_num):
                    aug_sentences = RS(org[self.sentence1], self.alpha, self.eda_num)
                    for s in aug_sentences:
                        org[self.sentence1] = s
                        out_validation_json.append(org.copy())
            else:
                for j in range(self.bt_num):
                    aug_sentences = RS(org[self.sentence1], self.alpha, self.eda_num)
                    for s in aug_sentences:
                        org[self.sentence1] = s
                        out_validation_json.append(org.copy())
                    aug_sentences = RS(org[self.sentence2], self.alpha, self.eda_num)
                    for s in aug_sentences:
                        org[self.sentence2] = s
                        out_validation_json.append(org.copy())
            
        with open(out_validation, 'w') as f:
            json.dump(out_validation_json, f, indent=2, ensure_ascii=False)

        del out_validation_json
        
        return True

    def random_deletion(self):
        outdir = self.output_dir + "_random_deletion/"
        if self.is_exists(outdir):
            return True

        pathlib.Path(outdir).mkdir(exist_ok=True, parents=True)
        out_train = outdir + "train.json"
        out_validation = outdir + "validation.json"

        out_train_json = []
        out_validation_json = []

        # Random Deletion: 어미 같은거 생각 안 함
        for k in range(len(self.dataset["train"][self.sentence1])):
            org = self.dataset["train"][k]
            if self.sentence2 == None:
                for j in range(self.bt_num):
                    aug_sentences = RD(org[self.sentence1], self.alpha, self.eda_num)
                    for s in aug_sentences:
                        org[self.sentence1] = s
                        out_train_json.append(org.copy())
            else:
                for j in range(self.bt_num):
                    aug_sentences = RD(org[self.sentence1], self.alpha, self.eda_num)
                    for s in aug_sentences:
                        org[self.sentence1] = s
                        out_train_json.append(org.copy())
                    aug_sentences = RD(org[self.sentence2], self.alpha, self.eda_num)
                    for s in aug_sentences:
                        org[self.sentence2] = s
                        out_train_json.append(org.copy())
            
        with open(out_train, 'w') as f:
            json.dump(out_train_json, f, indent=2, ensure_ascii=False)

        del out_train_json

        for k in range(len(self.dataset["validation"][self.sentence1])):
            org = self.dataset["validation"][k]
            if self.sentence2 == None:
                for j in range(self.bt_num):
                    aug_sentences = RD(org[self.sentence1], self.alpha, self.eda_num)
                    for s in aug_sentences:
                        org[self.sentence1] = s
                        out_validation_json.append(org.copy())
            else:
                for j in range(self.bt_num):
                    aug_sentences = RD(org[self.sentence1], self.alpha, self.eda_num)
                    for s in aug_sentences:
                        org[self.sentence1] = s
                        out_validation_json.append(org.copy())
                    aug_sentences = RD(org[self.sentence2], self.alpha, self.eda_num)
                    for s in aug_sentences:
                        org[self.sentence2] = s
                        out_validation_json.append(org.copy())
            
        with open(out_validation, 'w') as f:
            json.dump(out_validation_json, f, indent=2, ensure_ascii=False)

        del out_validation_json
        
        return True


    def easy_data_augmentation(self):
        # For the first time use wordnet
        #import nltk
        #nltk.download('wordnet')

        print("## Easy Data Augmentation Start ##")

        if self.flag_sr:
            if self.synonym_replacement():
                print("## Synonym Replacement Done ##")
        if self.flag_ri:
            if self.random_insertion():
                print("## Random Insertion Done ##")
        if self.flag_rs:
            if self.random_swap():
                print("## Random Swap Done ##")
        if self.flag_rd:
            if self.random_deletion():
                print("## Random Deletion Done ##")

        return True

    def bt_setup(self, tgt_lang, num):
        self.tgt_lang = tgt_lang
        self.bt_num = num