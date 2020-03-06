import numpy as np
import os
import re
#file_path="./en_cn/"
def get_lines(file_path):
    train_file=file_path+"train.txt"
    test_file=file_path+"test.txt"
    dev_file=file_path+"dev.txt"
    assert os.path.exists(train_file)==True==os.path.exists(test_file)==os.path.exists(dev_file)
    with open(train_file,'r',encoding='utf-8') as f:
        train_lines=f.readlines()
    with open(test_file,'r',encoding='utf-8') as f:
        test_lines=f.readlines()
    with open(dev_file,'r',encoding='utf-8') as f:
        dev_lines=f.readlines()
    lines=train_lines+test_lines+dev_lines
    assert type(lines)==list
    return lines#lines is a list, each element is a sentence like "It grew larger and larger. \t它变得越来越大。\n"

def split_en_cn(lines):
    en_sentence_list=[]
    cn_sentence_list=[]
    for line in lines:
        assert type(line)==str
        line_split=line.strip().split("\t")
        assert type(line_split)==list and len(line_split)==2
        en_sentence,cn_sentence=line_split[0],line_split[1]
        assert type(en_sentence)==type(cn_sentence)==str
        en_sentence_list.append(en_sentence)
        cn_sentence_list.append(cn_sentence)
    return en_sentence_list,cn_sentence_list

#split English sentence and Chinese sentence

def process_sentence_fn(en_sentence_list,cn_sentence_list):
    process_en_list=[]
    assert len(en_sentence_list)==len(cn_sentence_list)
    for each_en_sentence in en_sentence_list:
        lower_sentence=each_en_sentence.lower().strip()
        processed_sentence=re.sub(pattern=r"([,.!?])",repl=r" \1 ",string=lower_sentence)#seperate pountation and word use blank
        processed_sentence=re.sub(pattern=r"[^a-z?.,!]",repl=" ",string=processed_sentence)#The character except for a-z?.,! will be replaced by " "
        processed_sentence=re.sub(pattern=r"[' ']+",repl=" ",string=processed_sentence)
        processed_sentence=processed_sentence.rstrip().strip()
        
        processed_sentence="<start> "+processed_sentence+" <end>"
        process_en_list.append(processed_sentence)
    
    process_cn_list=[]
    for each_cn_sentence in cn_sentence_list:
        cn_sentence=" ".join([cn_word for cn_word in each_cn_sentence])
        cn_sentence="<start> "+cn_sentence+" <end>"
        #Because Chinese sentences have no definitly word or character, so we can take each character and each pountuation as a word
        process_cn_list.append(cn_sentence)

    assert len(process_en_list)==len(process_cn_list)
    en_list=[]
    cn_list=[]
    for process_en_sen,process_cn_sen in zip(process_en_list,process_cn_list):
        en_list.append(process_en_sen.strip().split())
        cn_list.append(process_cn_sen.strip().split())

    return en_list,cn_list

import operator
from collections import Counter

def get_word2id(en_list,cn_list):
    en_word2id={}
    cn_word2id={}
    en_word2id["<pad>"]=len(en_word2id)
    cn_word2id["<pad>"]=len(cn_word2id)
    all_en_words=[]
    all_cn_words=[]
    assert len(en_list)==len(cn_list)
    for en_sentence_list,cn_sentence_list in zip(en_list,cn_list):
        for each_en_word in en_sentence_list:
            all_en_words.append(each_en_word)
        for each_cn_word in cn_sentence_list:
            all_cn_words.append(each_cn_word)

    en_count=Counter(all_en_words)
    cn_count=Counter(all_cn_words)
    en_sorted=sorted(en_count.items(),key=operator.itemgetter(1),reverse=True)
    cn_sorted=sorted(cn_count.items(),key=operator.itemgetter(1),reverse=True)
    for en_word,freq in en_sorted:
        en_word2id[en_word]=len(en_word2id)
    for cn_word,freq in cn_sorted:
        cn_word2id[cn_word]=len(cn_word2id)

    cn_word2id["<unk>"]=len(cn_word2id)
    en_word2id["<unk>"]=len(en_word2id)
    return en_word2id,cn_word2id

def sorted_sentence(en_list,cn_list):
    sentence_length_dict={}
    for i in range(len(en_list)):
        sentence_length_dict[i]=len(en_list[i])
    sorted_sentence=sorted(sentence_length_dict.items(),key=operator.itemgetter(1))
    record_sentence_index=[]
    for sentence_index,sentence_length in sorted_sentence:
        record_sentence_index.append(sentence_index)
    return np.array(en_list)[record_sentence_index],np.array(cn_list)[record_sentence_index]

def sentence_to_id(sorted_en_list,sorted_cn_list,en_word2id,cn_word2id):
    en_sentence_list=sorted_en_list
    cn_sentence_list=sorted_cn_list

    en_id_list=[]
    cn_id_list=[]
    for en_sentence,cn_sentence in zip(en_sentence_list,cn_sentence_list):
        en_id_list.append([en_word2id.get(en_word,en_word2id["<unk>"]) for en_word in en_sentence])
        cn_id_list.append([cn_word2id.get(cn_word,cn_word2id["<unk>"]) for cn_word in cn_sentence])
    return en_id_list,cn_id_list

def review_en_sentence(en_id_l,en_word2id):
    en_id2word={k:v for v,k in en_word2id.items()}
    return " ".join([en_id2word.get(id_,"<unk>") for id_ in en_id_l])
def review_cn_sentence(cn_id_l,cn_word2id):
    cn_id2word={k:v for v,k in cn_word2id.items()}
    return " ".join([cn_id2word.get(id_,"<unk>") for id_ in cn_id_l])


