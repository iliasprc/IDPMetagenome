import argparse
import subprocess

from idp_programs.utils import select_method, post_process_seg_output, post_process_cast_output,metrics_seg,read_disprot

# paths to data
testdata_path = 'data/prion.fasta'
swiss_prot_data = 'data/SwissProt_uniprot-metagenome-filtered-reviewed yes.fasta'
disprot_data = 'data/DisProt release_2021_08.fasta'
caid_data = 'data/CAID_data_2018/fasta_files/data_disprot-disorder.txt'


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', type=str, default='seg', help='select method for detecting IDP regions',
                        choices=('cast', 'seg', 'iupred2a'))
    parser.add_argument('--log_interval', type=int, default=1000, help='steps to log.info metrics and loss')
    parser.add_argument('--dataset_name', type=str, default="SwissProt", help='dataset name',
                        choices=('SwissProt', 'DisProt', 'testdata_path', 'CAID2018'))
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
    ground_truth = swiss_prot_data
    method_args.insert(1, swiss_prot_data)
elif args.dataset_name == 'DisProt':
    ground_truth = disprot_data
    method_args.insert(1, disprot_data)
elif args.dataset_name == 'CAID2018':
    method_args.insert(1, caid_data)
results_file = f'{args.save}/{args.method}/{args.dataset_name}_seg_out.txt'
print(f' IDP METHOD ARGUMENTS\n{method_args}')
#subprocess.run(method_args)

with open(results_file, "w") as outfile:
    subprocess.run(method_args, stdout=outfile)
print('Finished')
if args.method == 'cast':

    if args.dataset_name == 'SwissProt':
        #    idp_count = post_process_seg_output(results_file,ground_truth)
        annotations = read_disprot(ground_truth)
        metrics_seg(results_file, annotations)
elif args.method == 'seg':
    if args.dataset_name == 'SwissProt':
    #    idp_count = post_process_seg_output(results_file,ground_truth)
        annotations = read_disprot(ground_truth)
        metrics_seg(results_file,annotations)
    #print(f'Number of IDP regions {idp_count} extracted from {args.method}')


