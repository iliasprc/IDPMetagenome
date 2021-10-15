import argparse
import subprocess

from idp_programs.utils import select_method, post_process_seg_output, post_process_cast_output

# paths to data
testdata_path = 'data/prion.fasta'
swiss_prot_data = 'data/SwissProt_uniprot-metagenome-filtered-reviewed yes.fasta'
disprot_data = 'data/DisProt release_2021_08.fasta'
caid_data = 'data/CAID_data_2018/fasta_files/data_disprot-disorder.txt'


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', type=str, default='cast', help='select method for detecting IDP regions',
                        choices=('cast', 'seg', 'iupred2a'))
    parser.add_argument('--log_interval', type=int, default=1000, help='steps to log.info metrics and loss')
    parser.add_argument('--dataset_name', type=str, default="CAID2018", help='dataset name',
                        choices=('SwissProt', 'DisProt', 'testdata_path', 'CAID2018'))

    # parser.add_argument('--tensorboard', action='store_true', default=True)

    parser.add_argument('--root_path', type=str, default='./data/data',
                        help='path to dataset ')
    parser.add_argument('--save', type=str, default='./results',
                        help='path to checkpoint save directory ')
    args = parser.parse_args()

    return args


args = arguments()

print(f' RUN ARGS \n{args.__dict__}')
method_args = select_method(args.method)
if args.dataset_name == 'SwissProt':
    method_args.insert(1, swiss_prot_data)
elif args.dataset_name == 'DisProt':
    method_args.insert(1, disprot_data)
elif args.dataset_name == 'CAID2018':
    method_args.insert(1, caid_data)
results_file = f'{args.save}/{args.method}/{args.dataset_name}_out.txt'
print(f' IDP METHOD ARGUMENTS\n{method_args}')
with open(results_file, "w") as outfile:
    subprocess.run(method_args, stdout=outfile)
print('Finished')
if args.method == 'cast':
    idp_count = post_process_cast_output(results_file)
elif args.method == 'seg':
    idp_count = post_process_seg_output(results_file)
print(f'Number of IDP regions {idp_count} extracted from {args.method}')

# subprocess.run(method_args)
