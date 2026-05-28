import argparse


__all__ = ['parse_args']


def parse_args():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument("--gpu", default='1', type=str, metavar='G',
                        help='GPU device number')
    
    # dataset and dataloader hyperparameters
    parser.add_argument('--dataset', default='mvtec', type=str, metavar='D',
                        help='dataset name: mvtec/btad/visa/isbi2016 (default: mvtec)')
    parser.add_argument('--test_dataset', default=None, type=str, metavar='D',
                        help='optional test dataset name for cross-domain evaluation (defaults to --dataset)')
    parser.add_argument('--data_path', default='./datasets/mvtec_anomaly_detection', type=str)
    parser.add_argument('--test_data_path', default=None, type=str,
                        help='optional test dataset root (defaults to --data_path)')
    parser.add_argument('--class_name', default='none', type=str, metavar='C',
                        help='class name for MVTecAD (default: none)')
    parser.add_argument('--test_class_name', default=None, type=str, metavar='C',
                        help='optional test class name for cross-domain evaluation')
    parser.add_argument('--inp_size', default=256, type=int, metavar='C',
                        help='image resize dimensions (default: 256)')
    parser.add_argument('--batch_size', default=32, type=int, metavar='B',
                        help='train batch size (default: 32)')
    parser.add_argument('--num_workers', default=4, type=int, metavar='G',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--data_strategy', default='0,1', type=str,
                        help='0: Repeated Utilization; 1: RandAugmented CutPaste; 2: RandAugmented CutPaste-Pseudo')
    parser.add_argument('--num_anomalies', default=5, type=int, metavar='G',
                        help='number of anomalies per category (default: 5)')
    parser.add_argument('--anomaly_source_path', default='', type=str,
                        help='Path of anomaly source dataset, used in RandAugmented CutPaste-Pseduo')
    parser.add_argument('--pseudo_type', default='nsa', type=str,
                        help='Pseudo anomaly type: nsa (Normal-to-Normal Patch Exchange) or perlin (texture mixing with Perlin mask)')
    parser.add_argument('--in_fg_region', default=True, type=bool,
                        help='Wether to restrict the pasting location in the foreground region of the normal samples')
    parser.add_argument('--prioritized_ano_types', default='', type=str,
                        help='Comma-separated list of subtype:multiplier to upweight anomaly subtypes, e.g. "thread_side:3,manipulated_front:2"')
    parser.add_argument('--use_seamless_clone', action='store_true', default=False,
                        help='Use OpenCV seamlessClone when pasting augmented region (slower, more realistic)')
    parser.add_argument('--strong_ops', type=int, default=6,
                        help='Number of random augment operations to use for prioritized subtypes (default: 6)')
    parser.add_argument('--placement_attempts', type=int, default=5,
                        help='Number of attempts to place a shifted anomaly into the foreground region before fallback (default: 5)')
    parser.add_argument('--balanced_data_loader', default=False, type=bool,
                        help='Balancing the number of normal and anomalous samples in a mini-batch')
    parser.add_argument('--steps_per_epoch', default=15, type=int,
                        help='Sampling batches per epoch')

    # model hyperparameter
    parser.add_argument('--backbone_arch', default='tf_efficientnet_b6', type=str, metavar='A',
                        help='feature extractor: (default: efficientnet_b6)')
    parser.add_argument('--backbone_base', default='resnet50', type=str, metavar='A',
                        help='base architecture for resnet_se backbone: resnet18/resnet34/resnet50 (default: resnet50)')
    parser.add_argument('--flow_arch', default='conditional_flow_model', type=str, metavar='A',
                        help='normalizing flow model (default: cnflow)')
    parser.add_argument('--feature_levels', default=3, type=int, metavar='L',
                        help='nudmber of feature layers (default: 3)')
    parser.add_argument('--coupling_layers', default=8, type=int, metavar='L',
                        help='number of coupling layers used in normalizing flow (default: 8)')
    parser.add_argument('--clamp_alpha', default=1.9, type=float, metavar='L',
                        help='clamp alpha hyperparameter in normalizing flow (default: 1.9)')
    parser.add_argument('--pos_embed_dim', default=128, type=int, metavar='L',
                        help='dimension of positional enconding (default: 128)')
    parser.add_argument('--pos_beta', default=0.05, type=float, metavar='L',
                        help='position hyperparameter for bg-sppc (default: 0.01)')
    parser.add_argument('--margin_tau', default=0.1, type=float, metavar='L',
                        help='margin hyperparameter for bg-sppc (default: 0.1)')
    parser.add_argument('--tau_min', default=0.02, type=float, metavar='L',
                        help='minimum margin tau (τ_min) for curriculum scheduling (default: 0.02)')
    parser.add_argument('--tau_max', default=0.1, type=float, metavar='L',
                        help='maximum/initial margin tau (τ_max) for curriculum scheduling (default: 0.1)')
    parser.add_argument('--tau_lambda', default=5.0, type=float, metavar='L',
                        help='decay rate λ used in τ(t)=τ_min+(τ_max-τ_min)*exp(-λ * t / T) (default: 5.0)')
    parser.add_argument('--normalizer', default=10, type=float, metavar='L',
                        help='normalizer hyperparameter for bg-sppc (default: 10)')
    parser.add_argument('--bgspp_lambda', default=1, type=float, metavar='L',
                        help='loss weight lambda for bg-sppc (default: 1)')
    parser.add_argument('--focal_weighting', action='store_true', default=False,
                        help='asymmetric focal weighting (default: False)')
    parser.add_argument('--ub_weight', default=1.0, type=float,
                        help='weight for UncertaintyBoundary loss (default: 1.0)')
    parser.add_argument('--ub_max_grad_clip', default=1.0, type=float,
                        help='max gradient clip for UncertaintyBoundary params (default: 1.0)')
    
    # learning hyperparamters
    parser.add_argument('--lr', type=float, default=2e-4, metavar='LR',
                        help='learning rate (default: 2e-4)')
    parser.add_argument('--lr_decay_epochs', nargs='+', default=[50, 75, 90],
                        help='learning rate decay epochs (default: [50, 75, 90])')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1, metavar='LR',
                        help='learning rate decay rate (default: 0.1)')
    parser.add_argument('--lr_warm', type=bool, default=True, metavar='LR',
                        help='learning rate warm up (default: True)')
    parser.add_argument('--lr_warm_epochs', type=int, default=2, metavar='LR',
                        help='learning rate warm up epochs (default: 2)')
    parser.add_argument('--lr_cosine', type=bool, default=True, metavar='LR',
                        help='cosine learning rate schedular (default: True)')
    parser.add_argument('--temp', type=float, default=0.5, metavar='LR',
                        help='temp of cosine learning rate schedular (default: 0.5)')                    
    parser.add_argument('--meta_epochs', type=int, default=25, metavar='N',
                        help='number of meta epochs to train (default: 25)')
    parser.add_argument('--sub_epochs', type=int, default=8, metavar='N',
                        help='number of sub epochs to train (default: 8)')

    # saving hyperparamters
    parser.add_argument('--output_dir', default='output', type=str, metavar='C',
                        help='name of the run (default: output)')
    parser.add_argument('--exp_name', default='bgad_fas', type=str, metavar='C',
                        help='name of the run (default: 0)')
    parser.add_argument('--checkpoint', default='', type=str, metavar='D',
                        help='used in test phase, set same with the exp_name')
    
    # misc hyperparamters
    parser.add_argument("--phase", default='train', type=str, metavar='T',
                        help='train or test phase (default: train)')
    parser.add_argument("--print_freq", default=2, type=int, metavar='T',
                        help='print frequency (default: 2)')                    
    parser.add_argument('--pro', action='store_true', default=False,
                        help='enables estimation of AUPRO metric')
    parser.add_argument('--vis', action='store_true', default=False,
                        help='test data visualizations')
    parser.add_argument('--with_fas', action='store_true', default=True,
                        help='Wether to train with few abnormal samples (default: True)')
    parser.add_argument('--without_fas', dest='with_fas', action='store_false',
                        help='Disable FAS branch and use bgad_train_engine (enables UB stats csv logging)')
    parser.add_argument('--disable_dfrm', action='store_true', default=False,
                        help='Disable DFRM (SE channel refinement) in backbone for ablation')
    parser.add_argument('--img_level', action='store_true', default=False,
                        help='Wether to train only on image-level features (default: False)')
    parser.add_argument('--not_in_test', action='store_true', default=True,
                        help='Wether to exclude the trained anomalies outside the testset (default: True)')
    parser.add_argument('--measure_inference', action='store_true', default=False,
                        help='Measure inference latency during validation (default: False)')
    parser.add_argument('--latency_skip', type=int, default=2,
                        help='Number of warmup batches to skip when reporting latency (default: 2)')
    # logging options
    parser.add_argument('--log_dir', default=None, type=str,
                        help='directory to save logs (default: <output_dir>/<exp_name>/logs)')
    parser.add_argument('--log_file', default=None, type=str,
                        help='explicit log file path (default: <log_dir>/<exp_name>.log)')
    parser.add_argument('--log_level', default='INFO', type=str,
                        help='logging level (DEBUG/INFO/WARNING/ERROR, default: INFO)')
    args = parser.parse_args()
    
    return args
