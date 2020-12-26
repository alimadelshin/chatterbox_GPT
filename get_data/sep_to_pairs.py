import time
import unicodedata 
import os
import numpy as np
import re

path = 'place/2019-'

def unicode_to_ascii(s):
    return ''.join( c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn')

def preprocess_sentence(w):
    w = unicode_to_ascii(w)

    w = re.sub(r"([?.!,])", r" \1 ", w)
    w = re.sub(r'[" "]+', " ", w)
    w = re.sub(r"[^a-zA-Z0-9?.!,]+", " ", w)
    w = w.rstrip().strip().lower()

    return w

miss_pairs = 0

lines = []

counter = 0
max_len = 256

f0 = open('unp_seq_inputs', 'a', encoding = 'UTF-8')
f1 = open('unp_seq_targets', 'a', encoding = 'UTF-8')
f2 = open('unp_scores', 'a', encoding= 'UTF-8')

for month in range(1, 13):
    with open(path + str(month), 'r', encoding = 'UTF-8', errors = 'ignore') as f:
        for line in f:
            if line == '<end examples>\n':
                if len(lines) > 1:
                    for r in range(len(lines) - 1):
                        quest = preprocess_sentence(' '.join(lines[r].split()[:-1] + '|<end_of_text>|' + lines[r + 1].split()[:-1]))
                        reply = preprocess_sentence(' '.join(lines[r + 1].split()[:-1] + '|<end_of_text>|'))

                        if len(quest.split()) + len(reply.split()) < max_len:
                            f0.write(quest + '\n')
                            f1.write(reply + '\n')
                            f2.write(lines[r + 1].split()[-1] +'\n')
                             
                        else:
                            miss_pairs += 1

                        counter += 1

                    if counter%100000 == 0:
                       print(counter)
            else:
                lines.append(line)






