import torch

from torch.utils.data import Dataset, DataLoader
from torchtext.data.functional import load_sp_model

from util.prepro import CustomDataset, get_samples, data_generation

import torch.nn.functional as F
import torch.distributed as dist
import torch.multiprocessing as mp
import numpy as np
import os
import argparse
import pickle
import datetime

from GPT2 import model

parser = argparse.ArgumentParser(description='GPT2')
parser.add_argument('--resume', default=None, help='')
parser.add_argument('--batch_size', type=int, default=64, help='')
parser.add_argument("--gpu_devices", type=int, nargs='+', default=None, help="")
parser.add_argument('--gpu', default=None, type=int, help='GPU id to use.')
parser.add_argument('--accumulate_steps', default = 1, type= int, help = 'Accumulated steps.')
args = parser.parse_args()

gpu_devices = ','.join([str(id) for id in args.gpu_devices])
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_devices

class Schedule:
    def __init__(self, d_model, warmup_steps = 4000):
        self.d_model = torch.tensor(d_model, dtype= torch.float32)
        self.warmup_steps = torch.tensor(warmup_steps, dtype= torch.float32)

    def next_step(self, step):
        step = torch.tensor(step, dtype= torch.float32)

        arg_1 = torch.rsqrt(step)
        arg_2 = step * (self.warmup_steps ** -1.5)

        return torch.rsqrt(self.d_model) *  torch.min(arg_1, arg_2)
        

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(device)    
    encoder = load_sp_model('util/bpe.model')
    print('Tokenizer loaded..')

    num_layers = 32
    d_model = 1024
    num_heads =  8
    dropout_rate = 0.1

    print('==> Making model..')    
    net = model.GTP2(num_layers = num_layers, d_model = d_model, num_heads = num_heads, vocab_size=len(encoder), rate=dropout_rate)

    for p in net.parameters():
        if p.dim() > 1:
            torch.nn.init.xavier_uniform_(p)

    net.cuda(device)

    num_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print('The number of parameters of model is', num_params)
    
    scheduler = Schedule(d_model)
    optimizer = torch.optim.Adam(net.parameters(), lr=scheduler.next_step(step=1))

    train(net, encoder, optimizer, scheduler, device, args.accumulate_steps)

def train(net, encoder,  optimizer, scheduler,device, accumulate_steps): 
    z = open('logger', 'a')

    net.train()
    
    total_loss = 0.
    batches_done = 0
    p_size = 3840
    
    mem_steps = accumulate_steps
    

    for i in range(0,11):
        data = data_generation(f'data/q{i}.txt',f'data/r{i}.txt')
        
        print('steps_left:', ((11 - i) * len(data))/(mem_steps* args.batch_size))
        
        print(f'files number: {i}')

        for p in range(len(data)//p_size - 1):

            inputs, targets = get_samples(encoder, data[p*p_size:(p + 1)*p_size])
        
            dataset_train = CustomDataset(inputs, targets)            
            train_loader = DataLoader(dataset_train, batch_size=args.batch_size, num_workers=0)
            step = 0

            
            for inputs, targets in train_loader:
                if  step == 0:
                    optimizer.zero_grad()

                inputs = inputs.cuda(device)
                targets = targets.cuda(device)

                size = inputs.size(1)
               
                input_mask = (inputs != 0).unsqueeze(1)
                nopeak_mask = np.triu(np.ones((1, size, size)), k=1)
                nopeak_mask = torch.autograd.Variable(torch.from_numpy(nopeak_mask) == 0).cuda(device)
                
                input_mask = input_mask & nopeak_mask
        
                outputs = net(inputs, input_mask)

                loss = F.cross_entropy(outputs.view(-1, outputs.size(-1)), targets.contiguous().view(-1),ignore_index=0)


                loss.mean().backward()                
                total_loss += loss.data                

                if step == (mem_steps - 1):
                    optimizer.step()
                    step = 0
                    batches_done  += 1 
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = scheduler.next_step(batches_done)

                    if batches_done % 200  == 0:
                        total_loss /= mem_steps
                        print(f'batch_idx: {(batches_done)} loss: {total_loss/200} time: {datetime.datetime.now().strftime("%H:%M:%S")} lr: {param_group["lr"]}')
                        z.write(f'batch_idx: {(batches_done)} loss: {total_loss/200} time: {datetime.datetime.now().strftime("%H:%M:%S")}\n')
                        total_loss = 0
                    if batches_done  % 2500 == 0:   
                            torch.save(net.state_dict(), f'checkpoints/checkpoint_batch_id_{batches_done}')

                else:
                    step += 1





if __name__=='__main__':
    main()