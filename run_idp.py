import argparse
import subprocess
from idp_programs.arguments import select_method
from idp_programs.utils import  metrics_seg, read_disprot, cast_metrics, iupred2a_metrics

# paths to data
testdata_path = 'data/TEST.fasta'
swiss_prot_data = 'data/SwissProt_uniprot-metagenome-filtered-reviewed yes.fasta'
disprot_data = 'data/DisProt release_2021_08.fasta'
caid_data = 'data/CAID_data_2018/fasta_files/data_disprot-disorder.txt'
mxd494 = 'data/idp_seq_2_seq/mxd494/data_MXD494.txt'
disorder723 = 'data/idp_seq_2_seq/disorder723/data_disorder723.txt'
dm = 'data/idp_seq_2_seq/validation/data_all_valid.txt'
def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', type=str, default='seg', help='select method for detecting IDP regions',
                        choices=('cast', 'seg', 'iupred2a'))
    parser.add_argument('--log_interval', type=int, default=1000, help='steps to log.info metrics and loss')
    parser.add_argument('--dataset_name', type=str, default="testdata_path", help='dataset name',
                        choices=('SwissProt', 'DisProt', 'testdata_path', 'CAID2018', 'MXD494','disorder723','dm'))
    parser.add_argument('--root_path', type=str, default='./data/data',
                        help='path to dataset ')
    parser.add_argument('--save', type=str, default='./results',
                        help='path to checkpoint save directory ')
    parser.add_argument('--run_metrics',  default=False,
                        help=' ')
    args = parser.parse_args()

    return args


args = arguments()

print(f'RUN ARGS \n{args.__dict__}')
method_args = select_method(args.method)
if args.dataset_name == 'SwissProt':
    ground_truth = swiss_prot_data
    method_args.insert(1, swiss_prot_data)
elif args.dataset_name == 'DisProt':
    ground_truth = disprot_data
    method_args.insert(1, disprot_data)
elif args.dataset_name == 'CAID2018':
    ground_truth = './data/CAID_data_2018/annotation_files/annot_disprot-disorder.txt'
    method_args.insert(1, caid_data)
elif args.dataset_name == 'MXD494':
    method_args.insert(1, mxd494)
elif args.dataset_name == 'disorder723':
    method_args.insert(1, disorder723)
elif args.dataset_name == 'dm':
    method_args.insert(2, dm)
else:
    method_args.insert(1,testdata_path)
results_file = f'{args.save}/{args.method}/{args.dataset_name}__out.txt'
print(f' IDP METHOD ARGUMENTS\n{method_args}')
#subprocess.run(method_args)
#exit()
with open(results_file, "w") as outfile:
    subprocess.run(method_args, stdout=outfile)
print('Finished')

if args.run_metrics:
    if args.method == 'cast':

        if args.dataset_name == 'SwissProt':
            #    idp_count = post_process_seg_output(results_file,ground_truth)
            annotations = read_disprot(ground_truth)
            cast_metrics(results_file, annotations)
        elif args.dataset_name == 'MXD494':
            # annotations = read_disprot('/home/iliask/PycharmProjects/MScThesis/data/idp_seq_2_seq/mxd494/annot_MXD494
            # .txt')
            ground_truth = '/home/iliask/PycharmProjects/MScThesis/data/idp_seq_2_seq/mxd494/annot_MXD494.txt'
            cast_metrics(results_file, ground_truth)
        elif args.dataset_name == 'disorder723':
            # annotations = read_disprot('/home/iliask/PycharmProjects/MScThesis/data/idp_seq_2_seq/mxd494/annot_MXD494
            # .txt')
            ground_truth = '/home/iliask/PycharmProjects/MScThesis/data/idp_seq_2_seq/disorder723/annot_disorder723.txt'
            cast_metrics(results_file, ground_truth)
        elif args.dataset_name == 'DisProt':
            annotations = read_disprot(ground_truth)
            cast_metrics(results_file, annotations)
        elif args.dataset_name == 'CAID2018':
            # annotations = read_disprot(ground_truth)

            cast_metrics(results_file, ground_truth)
    elif args.method == 'seg':
        if args.dataset_name == 'SwissProt':
            #    idp_count = post_process_seg_output(results_file,ground_truth)
            annotations = read_disprot(ground_truth)
            metrics_seg(results_file, annotations)
        elif args.dataset_name == 'MXD494':
            # annotations = read_disprot('/home/iliask/PycharmProjects/MScThesis/data/idp_seq_2_seq/mxd494/annot_MXD494
            # .txt')
            ground_truth = '/home/iliask/PycharmProjects/MScThesis/data/idp_seq_2_seq/mxd494/annot_MXD494.txt'
            annotations = read_disprot(ground_truth)
            metrics_seg(results_file, annotations)
        elif   args.dataset_name == 'disorder723':
            ground_truth = '/home/iliask/PycharmProjects/MScThesis/data/idp_seq_2_seq/disorder723/annot_disorder723.txt'
            annotations = read_disprot(ground_truth)
            metrics_seg(results_file, annotations)
        elif args.dataset_name == 'DisProt':
            annotations = read_disprot(ground_truth)
            metrics_seg(results_file, annotations)
        elif args.dataset_name == 'CAID2018':
            # annotations = read_disprot(ground_truth)

            annotations = read_disprot(ground_truth)
            metrics_seg(results_file, annotations)
        # print(f'Number of IDP regions {idp_count} extracted from {args.method}')
    elif args.method == 'iupred2a':
        if args.dataset_name == 'SwissProt':
            #    idp_count = post_process_seg_output(results_file,ground_truth)
            annotations = read_disprot(ground_truth)
            metrics_seg(results_file, annotations)
        elif args.dataset_name == 'MXD494':
            # annotations = read_disprot('/home/iliask/PycharmProjects/MScThesis/data/idp_seq_2_seq/mxd494/annot_MXD494
            # .txt')
            ground_truth = '/home/iliask/PycharmProjects/MScThesis/data/idp_seq_2_seq/mxd494/annot_MXD494.txt'
            annotations = read_disprot(ground_truth)
            iupred2a_metrics(results_file, annotations)
        elif args.dataset_name == 'disorder723':
            ground_truth = '/home/iliask/PycharmProjects/MScThesis/data/idp_seq_2_seq/disorder723/annot_disorder723.txt'
            annotations = read_disprot(ground_truth)
            iupred2a_metrics(results_file, annotations)
        elif args.dataset_name == 'DisProt':
            annotations = read_disprot(ground_truth)
            iupred2a_metrics(results_file, annotations)
        elif args.dataset_name == 'CAID2018':
            # annotations = read_disprot(ground_truth)

            annotations = read_disprot(ground_truth)
            iupred2a_metrics(results_file, annotations)
        elif args.dataset_name == 'dm':
            ground_truth = 'data/idp_seq_2_seq/validation/annot_all_valid.txt'
            annotations = read_disprot(ground_truth)
            iupred2a_metrics(results_file, annotations)