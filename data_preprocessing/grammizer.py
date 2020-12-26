import pickle


def pop2d(path, limit_value = 800, buffer = 1500000):

    p1d = dict()
    p2d = dict()
    
    rows1d = 0
    rows2d = 0

    with open(path, 'r', encoding = 'UTF-8') as f:
        for line in f:
            tokens = line.lower().rstrip().replace('.', ' ').replace('?', ' ').replace('!', ' ').replace(',', ' ').split()

            if len(tokens) == 1:
                if p1d.get(tokens[0]):
                    p1d[tokens[0]] += 1
                else:
                    p1d[tokens[0]] = 1
                
                rows1d += 1
                
                if rows1d % buffer == 0:
                    temp = p1d.copy()
                    for key, value in temp.items():
                        if value < limit_value:
                            del p1d[key]
                    del temp 
                
                    print(rows1d)
            
            elif len(tokens) == 2:
                tokens = tuple(tokens)
                if p2d.get(tokens):
                    p2d[tokens] += 1
                else:
                    p2d[tokens] = 1

                rows2d += 1
                
                if rows2d % buffer == 0:
                    temp = p2d.copy()
                    for key, value in temp.items():
                        if value < limit_value:
                            del p2d[key] 
                    del temp
                
                    print(rows2d)

        for i, n_dict in enumerate([p1d, p2d]):
            temp = n_dict.copy()
            for key, value in temp.items():
                if value < limit_value:
                    del n_dict[key]
            
            with open(str(i + 1) + '_gramm' + '.pickle', 'wb') as handle:
                pickle.dump(n_dict, handle)


def grammer(path, limit_value = 1500, buffer = 1500000):

    n_grams = dict()
    n_size = 3
    rows = 0
    
    with open(path , 'r', encoding = 'UTF-8') as f:
        for line in f:
            tokens = line.lower().rstrip().replace('.', ' ').replace('?', ' ').replace('!', ' ').replace(',', ' ').split()
            
            if len(tokens) >= n_size:
                grams = zip(*[tokens[i:] for i in range(n_size)])
                for gram in grams:
                    if n_grams.get(gram):
                        n_grams[gram] += 1
                    else:
                        n_grams[gram] = 1
    
            rows += 1
    
            if rows % buffer == 0:
    
                temp  = {}
                values = list(n_grams.values())
                values.sort(reverse=True)
                values = values[:25000]

                
                for key, value in n_grams.items():
                    if value in values:
                        temp[key] = value
                        del values[values.index(value)]
                        
                        if not values:
                            n_grams = temp
                            del temp
                            break 
                
                print(rows)
    
    temp  = {}
        
    for key, value in n_grams.items():
        if value > limit_value:
            temp[key] = value
    
    with open(str(n_size) + '_gramm' + '_.pickle', 'wb') as handle:
        pickle.dump(temp, handle)
        print(str(n_size) + '_gramm' + '.pickle' + ' created...')


if __name__ == "__main__":
    pop2d('data/unpqueries')
    grammer('data/unpqueries')





