import pickle


def norm(path_inp, path_tar, path_score, limit = 0.6, delete = 1500):
    n_gramms = dict()

    for i in range(1,4):
        with open('gramms/' + str(i) + '_gramm' + '_' + '.pickle', 'rb') as handle:
            n_gramms[str(i) + '_gramm'] = pickle.load(handle)
    
    def reduction(tokens):

        if len(tokens) == 1:
            key = str(1) + '_gramm'
            token = tokens[0]
            if n_gramms[key].get(token):
                n_gramms[key][token] -= 1
                if n_gramms[key][token] < 800:
                    del n_gramms[key][token]
                return False
            else:
                return True
        
        elif len(tokens) == 2:
            key = str(2) + '_gramm'
            tokens = tuple(tokens)
            if n_gramms[key].get(tokens):
                n_gramms[key][tokens] -= 1
                if n_gramms[key][tokens] < 800:
                    del n_gramms[key][tokens]
                return False
            else:
                return True
            

    def cleanin(line):
        tokens = line.strip().replace('?', ' ').replace('!', ' ').replace('.', ' ').split()
        num = 3

        if len(tokens) == 0:
            return False
        
        if len(tokens) == 1 or len(tokens) == 2:
            return reduction(tokens)


        block = [0 for _ in tokens]
        n_gramms_del = []
        
         
        if len(tokens) >= num: 
            temp_gramms = zip(*[tokens[x:] for x in range(num)])
            for i, gram in enumerate(temp_gramms):
                if n_gramms[str(num) + '_gramm'].get(gram):
                    block[i : i + num] = [1] * num
                    n_gramms_del.append((str(num) + '_gramm', gram))
        

        if (1 - block.count(1)/len(block)) > limit:            
            for key_0, key_1 in n_gramms_del:
                try:
                    if n_grams[key_0][key_1] < delete:
                        del n_gramms[key_0][key_1]
    
                    else:
                        n_gramms[key_0][key_1] -= 1
                except:
                    pass
            return True
        else:
            return False
        
    inp = open('u_queries', 'w')
    tar = open('u_replies', 'w')
    score = open('u_score', 'w')

    all_lines = 0
    skipped_lines = 0

    with open(path_inp, 'r') as f:
        with open(path_tar, 'r') as g:
            with open(path_score, 'r') as z:
                for (i, (temp_inp, temp_tar, temp_score)) in enumerate(zip(f, g, z)):
                    if i % 10000000 == 0:
                        print(i)
                    if cleanin(temp_inp):
                        inp.write(temp_inp)
                        tar.write(temp_tar)
                        score.write(temp_score)
    
                    else:
                        skipped_lines += 1
                    
                    all_lines = i
                
                print(limit, ' : ', skipped_lines/all_lines)


if __name__ == "__main__":
    norm('data/unpqueries', 'data/unpreplies', 'data/scores')

                
 

         

