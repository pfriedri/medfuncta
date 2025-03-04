import yaml
import os.path
from argparse import ArgumentParser


def load_cfg(args):
    with open(args.config, "rb") as f:
        cfg = yaml.safe_load(f)

    for key, value in cfg.items():
        args.__dict__[key] = value

    return args


def parse_args():
    parser = ArgumentParser()

    """ Config """
    parser.add_argument('--config', help='Path to config .yaml file', type=str, default=None)

    """ System configuration """
    parser.add_argument('--gpu_id', help='GPU ID', type=int, default=0)
    parser.add_argument('--seed', help='Random seed', type=int, default=42)

    """ Resume training """
    parser.add_argument('--resume_path', help='Path to the logdir of training to resume', type=str)

    """ Training configuration """
    parser.add_argument('--dataset', help='Dataset', type=str)
    parser.add_argument('--img_size', help='Image size', type=int, default=64)
    parser.add_argument('--batch_size', help='Batch size used for training', type=int, default=8)
    parser.add_argument('--outer_steps', help='Numer of meta-learning steps to perform', type=int, default=1000000)
    parser.add_argument('--inner_steps', help='Number of inner loop optimization steps (G)', type=int, default=10)
    parser.add_argument('--inner_steps_boot', help='Number of inner loop update steps for bootstrapping (optional)', type=int, default=0)
    parser.add_argument('--meta_lr', help='Learning rate for meta-learning updates (beta)', type=float, default=3e-6)
    parser.add_argument('--inner_lr', help='Learning rate for inner loop (alpha)', type=float, default=1e-2)
    parser.add_argument('--inner_lr_boot', help='Learning rate for inner loop bootstrapping (optional)', type=float, default=1e-2)
    parser.add_argument('--data_ratio', help='Ratio of data used for training (gamma)', type=float, default=0.25)
    parser.add_argument('--sample_type', help='Coordinate sampling type', type=str, choices=['random', 'gradncp'], default='random')
    parser.add_argument('--lam', help='Scaling parameter lambda for weighting bootstrapping loss (optional)', type=float, default=100.)

    """ Testing configuration """
    parser.add_argument('--test_batch_size', help='Batch size used for testing', type=int, default=8)
    parser.add_argument('--num_test_signals', help='Number of signals used for testing', default=128, type=int)
    parser.add_argument('--inner_steps_test', help='Number of inner loop update steps at test-time (H)', type=int, default=20)

    """ Model configuration """
    parser.add_argument('--hidden_size', help='MLP hidden size (L)', type=int, default=256)
    parser.add_argument('--num_layers', help='Number of MLP layers (K)', type=int, default=15)
    parser.add_argument('--latent_modulation_dim', help='Representation size (P)', type=int, default=2048)
    parser.add_argument('--w0', help='SIREN parameter w0 (if used with w0-schedule w0,1)', type=float, default=30.)
    parser.add_argument('--w0_increment', help='If > 0, w0 is increased by w0_increment per layer', type=float, default=0.)
    parser.add_argument('--modulate_shift', help='Set True to use shift modulations', type=eval, default=True)
    parser.add_argument('--modulate_scale', help='Set True to use scale modulations (not recommended)', type=eval, default=False)
    parser.add_argument('--enable_skip_connections', help='Set True to enable skip-connections', type=eval, default=True)

    """ Logging configuration """
    parser.add_argument('--print_step', help='Print every x steps', type=int, default=100)
    parser.add_argument('--eval_step', help='Evaluate every x steps', type=int, default=1000)
    parser.add_argument('--save_step', help='Save model every x steps', type=int, default=50000)

    """ Eval configuration """
    parser.add_argument('--load_path', help='Load model from this path', type=str, default=None)

    """ Fitting configuration """
    parser.add_argument('--save_dir', help='Directory to store shared model, modulations and labels', type=str, default=None)

    """ Parse Arguments """
    args = parser.parse_args()

    """ Load config files """
    if args.config is not None and os.path.exists(args.config):
        load_cfg(args)

    return args
