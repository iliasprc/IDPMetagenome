import numpy as np

def post_process_cast_output(path):
    with open(path, 'r') as file1:
        data = file1.readlines()
        print(len(data))
        count = 0
        protein_seq = ''
        protein_count = 0
        protein_ids = []
        proteins = []
        s = ''
        idpr = ''
        idp_regions = []
        for idx, i in enumerate(data):

            i = i.strip()

            if 'region' in i:
                count += 1

                print(i)
                # print(re.sub('\D', '_', i))
                region_start = int(i[18:].split('to')[0].replace(" ", ""))
                region_end = int(i.split('to')[-1].split('corrected')[0].replace(" ", ""))
                # print(i[18:],region_start)
                idpr += f"{region_start},{region_end},"
                print(f"{region_start},{region_end},")
            elif '>' in i:
                regions = []
                protein_ids.insert(protein_count, i)
                proteins.insert(protein_count, protein_seq)
                idp_regions.insert(protein_count, idpr)
                protein_count += 1
                print(i, protein_seq)
                protein_seq = ''
                idpr = ''
            else:
                protein_seq += i

        print(count)
        # while True:
        #     count += 1
        #
        #     # Get next line from file
        #     line = file1.readline()
        #
        #     # if line is empty
        #     # end of file is reached
        #     if not line:
        #         print('break')
        #         break
        #     print("Line{}: {}".format(count, line.strip()))
        proteins.append(protein_seq)
        proteins.pop(0)
        idp_regions.append(idpr)
        idp_regions.pop(0)
        print(len(proteins), len(protein_ids), len(idp_regions))
        print(protein_ids[-1], '\n', proteins[-1], '\n', idp_regions[-1])

        file1.close()
        with open(path + 'postprocessed.txt', 'w') as f:
            for i in range(len(proteins)):
                f.write(f"{protein_ids[i]}\n{proteins[i]}\n{idp_regions[i]}\n")
        f.close()
        for i in range(len(proteins)):
            s = (f"{protein_ids[i]}\n{proteins[i]}\n{idp_regions[i]}\n")
            sequence_len = len(proteins[i])
            print(idp_regions[i].split(','))
            regions = idp_regions[i].split(',')
            if len(regions) == 1:
                pred = np.zeros(sequence_len)
            else:
                pred = np.zeros(sequence_len)
                regions = regions[:-1]
                regions = [int(i) for i in regions]
                print(regions)
                regions = iter(regions)
                for x in regions:
                    start, end = x, next(regions)
                    pred[start:end] = 1
                print(sequence_len, pred)
            pred_string = ''
            pred = pred.tolist()
            for i in pred:
                pred_string += str(int(i))
            print(pred_string)

            # print(regions)

        return count