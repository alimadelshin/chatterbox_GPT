import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import lr_scheduler
import torch.nn.functional as F
import math
import pickle

from GPT2 import model
import time
import datetime
from torchtext.data.functional import load_sp_model
import numpy as np 
import random as rnd

import os
import argparse

from util.prepro import CustomDataset, get_samples

import time

check_path = ""

os.environ["CUDA_VISIBLE_DEVICES"] = '0'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

num_layers = 32
d_model = 1024
num_heads = 8
pe_max = 126

print('Tokenizer loading..')
encoder = load_sp_model('util/bpe.model')
print('Tokenizer loaded!')

net = model.GTP2(num_layers = num_layers, d_model = d_model, num_heads = num_heads, vocab_size=len(encoder), rate=0.1)

model = torch.load(check_path, map_location='cuda:0')
net.load_state_dict(model)

net.cuda()
net.eval()

for _ in range(5):
    query = input() + '|<end_of_text>|'
    print(f'Input: {query}') 
    print(8 * '~' + 'Outputs' + 8 * '~')
    for T in [0.1, 0.3, 0.7, 0.75, 0.85]:
        query_ = torch.tensor(encoder.encode_as_ids(query)).unsqueeze(0)
        length = len(query_[0])
        for i in range(pe_max - length):
            size = query_.size(1)

            input_mask = (query_ != 0).unsqueeze(1)
            nopeak_mask = np.triu(np.ones((1, size, size)), k=1)
            nopeak_mask = torch.autograd.Variable(torch.from_numpy(nopeak_mask) == 0).cuda(device)

            input_mask = input_mask.cuda(device) & nopeak_mask
              
            output = net(query_.cuda(device), input_mask)
            output = F.softmax(output/T, dim = -1)
    
            next_word = torch.multinomial(output[:, -1], num_samples = 1)
            
            if next_word.squeeze(0) == encoder.encode_as_ids('|<end_of_text>|'):
                break 
       
            query_ = torch.cat([query_.cuda(device), next_word.cuda(device)], 1)
        
        print('T = {} : {}'.format(T, encoder.decode_ids(query_.squeeze(0)[:].tolist())))
 
    
    
    print(19 * '~')
