def has_numbers(inputString):
    return any(char.isdigit() for char in inputString)
def remove_blanks(path):
    with open(path,'r') as f:
        data = f.read().splitlines()
    with open('/home/iliask/PycharmProjects/MScThesis/scratch.txt', 'w') as f1:
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


            if line.rstrip():
                #print(line,'\n')
                if ">" in line:
                    print(f'{sequence}\n{annot}\n{line}')
                    f1.write(f'{sequence}\n{annot}\n{line}\n')
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
        print(f'{sequence}\n{annot} ')

def create_annot_fasta(path):
    name = path.split('/')[-1]
    pathfolder,name = path.rsplit('/',1)
    name_annot= f'{pathfolder}/annot_{name}'
    print(name,path,name_annot)
    name_prot = f'{pathfolder}/data_{name}'
    names,annotations,proteins,classes,w2i= read_data_(path)
    with open(name_annot,'w') as f:
        for i in range(len(names)):
            f.write(f"{names[i]}\n{annotations[i]}\n")
    with open(name_prot,'w') as f:
        for i in range(len(names)):
            f.write(f"{names[i]}\n{proteins[i]}\n")

def read_data_(path):
    classes = []
    with open(path,'r') as f:
        data = f.read().splitlines()
    
        proteins = []
        names = []
        annotations = []
        sequence = ''
        annot = ''
        cou = 0
        c = 0
        for idx,line in enumerate(data):

            if line.rstrip():
              #  print(line)
                if ">" in line:
                    names.append(line)
                    c+=1
                elif not has_numbers(line):
                    cou+=1
                    for chari in line:
                        if chari not in classes:
                            classes.append(chari)
                    proteins.append(line)
                elif has_numbers(line):
                    annotations.append(line)
        print(len(names), len(proteins),len(annotations))
        assert len(names) == len(proteins),print(len(names),len(proteins))
        assert len(proteins) == len(annotations),print(len(annotations),len(proteins))
        classes = sorted(classes)
        indixes = list(range(len(classes)))
        #print(classes)
        w2i = dict(zip(classes,indixes))
       # print(w2i)
        print(len(classes))
        return names,annotations,proteins,classes,w2i

classes = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y']
print(len(classes))
# names,annotations,proteins,classes,w2i = read_data_('/home/iliask/PycharmProjects/MScThesis/data/CAID_data_2018/fasta_files/data_disprot-disorder.txt')
#names,annotations,proteins,classes,w2i = read_data_('/home/iliask/PycharmProjects/MScThesis/data/idp_seq_2_seq/mxd494/MXD494.txt')
# import torch
# for i in range(len(names)):
#
#     print(names[i],'\n',proteins[i],'\n',annotations[i])
#     x = [w2i[amino] for amino in  proteins[i]]
#     y = torch.FloatTensor([int(k) for k in annotations[i]])
#     print(y)
#
#remove_blanks('/home/iliask/PycharmProjects/MScThesis/data/idp_seq_2_seq/disorder723/disorder723.txt')
create_annot_fasta('/home/iliask/PycharmProjects/MScThesis/data/idp_seq_2_seq/disorder723/disorder723.txt')
