def has_numbers(inputString):
    return any(char.isdigit() for char in inputString)
def remove_blanks(path):
    with open(path,'r') as f:
        data = f.read().splitlines()
        proteins = []
        names = []
        annotations = []
        sequence = ''
        annot = ''
        for line in data:
            #print(i)
            # if ">" in i:
            # if "" in i:
            #     print(i)
            if '>DM_train280' in line:
                exit()
            if line.rstrip():
                #print(line,'\n')
                if ">" in line:
                    print(f'{sequence}\n{annot}\n{line}')
                    sequence  = ''
                    annot = ''
                if not has_numbers(line):
                    #print(sequence)
                    annot = ''
                    sequence+=line

                elif has_numbers(line):
                    annot+=line
                    #print(sequence)
            # else:
            #     print('not rstrip',line)


def read_data_(path):
    classes = []
    with open(path,'r') as f:
        data = f.read().splitlines()
        proteins = []
        names = []
        annotations = []
        sequence = ''
        annot = ''
        for line in data:
            if line.rstrip():
                #print(line,'\n')
                if ">" in line:
                    names.append(line)
                elif not has_numbers(line):
                    for chari in line:
                        if chari not in classes:
                            classes.append(chari)
                    proteins.append(line)
                elif has_numbers(line):
                    annotations.append(line)
        assert len(names) == len(proteins)
        classes = sorted(classes)
        indixes = list(range(len(classes)))
        #print(classes)
        w2i = dict(zip(classes,indixes))
        #print(w2i)
       # print(len(classes))
        return names,annotations,proteins,classes,w2i
# names,annotations,proteins,classes,w2i = read_data_('/home/iliask/PycharmProjects/MScThesis/data/idp_seq_2_seq/train/all_train.txt')
# import torch
# for i in range(len(names)):
#
#     print(names[i],'\n',proteins[i],'\n',annotations[i])
#     x = [w2i[amino] for amino in  proteins[i]]
#     y = torch.FloatTensor([int(k) for k in annotations[i]])
#     print(y)
#
# remove_blanks('/home/iliask/PycharmProjects/MScThesis/data/idp_seq_2_seq/validation/ldr_valid.txt')