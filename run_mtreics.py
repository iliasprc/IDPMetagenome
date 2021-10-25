from idp_programs.utils import  metrics_seg, read_disprot, cast_metrics, iupred2a_metrics

ground_truth = '/home/iliask/PycharmProjects/MScThesis/data/idp_seq_2_seq/mxd494/annot_MXD494.txt'
cast_metrics(results_file, ground_truth)