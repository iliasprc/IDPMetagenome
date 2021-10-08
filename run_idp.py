import argparse
import subprocess

from idp_programs.utils import arguments,select_method
# paths to data
testdata_path = 'data/prion.fasta'
swiss_prot_data = 'data/SwissProt_uniprot-metagenome-filtered-reviewed yes.fasta'
disprot_data='data/DisProt release_2021_08.fasta'

def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', type=str, default='cast', help='select method for detecting IDP regions',
                        choices=('cast', 'seg', 'iupred2a'))
    parser.add_argument('--log_interval', type=int, default=1000, help='steps to log.info metrics and loss')
    parser.add_argument('--dataset_name', type=str, default="DisProt", help='dataset name',choices=('SwissProt','DisProt','testdata_path'))

    parser.add_argument('--tensorboard', action='store_true', default=True)

    parser.add_argument('--root_path', type=str, default='./data/data',
                        help='path to dataset ')
    parser.add_argument('--save', type=str, default='./results',
                        help='path to checkpoint save directory ')
    args = parser.parse_args()
    print(args.__dict__)

    return args
args = arguments()
method_args =select_method(args.method)
if args.dataset_name == 'SwissProt':
    method_args.insert(1, swiss_prot_data)
elif args.dataset_name =='DisProt':
    method_args.insert(1, disprot_data)




results_file = f'{args.save}/{args.method}/{args.dataset_name}_out.txt'
print(method_args)
# with open(results_file, "w") as outfile:
#     subprocess.run(method_args, stdout=outfile)
subprocess.run(method_args)