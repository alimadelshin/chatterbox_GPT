import os
import random

i = 0

with open('data/pretraindatareal', 'r') as f:
    while f.readline():
        i += 1

part = i//14 - 1

print(f'size: {i}, part_size: {part}')

with open('data/queries', 'r') as f:
    with open('data/replies', 'r') as g:
        k = open('shuffled_queries', 'w')
        l = open('shuffled_replies', 'w')

        shuffle_block = []

        for i, block in enumerate(zip(f, g)):
            shuffle_block.append(block)
            
            if i % part == 0:
                print(len(shuffle_block))
                random.shuffle(shuffle_block)
                for q, r in shuffle_block:
                    k.write(q)
                    l.write(r)
                
                shuffle_block = []
        


                   

                   

