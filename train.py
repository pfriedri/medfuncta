import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from common.args import parse_args
from common.utils import set_random_seed, Logger, InfiniteSampler
from data.dataset import get_dataset
from models.inrs import LatentModulatedSIREN
from models.model_wrapper import ModelWrapper
from train.trainer import trainer
from train.maml_boot import train_step
from eval.maml_scale import test_model


def main(args):
    """
    Main function to call for running a training procedure.
    :param args: parameters parsed from the command line.
    :return: Nothing.
    """

    """ Set a device to use """
    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu_id)
    device = torch.device(f'cuda' if torch.cuda.is_available() else 'cpu')
    args.device = device

    """ Enable determinism """
    set_random_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

    """ Define dataset """
    train_set, test_set = get_dataset(args)

    """ Define dataloader """
    infinite_sampler = InfiniteSampler(train_set, rank=0, num_replicas=1, shuffle=True, seed=args.seed)
    train_loader = DataLoader(train_set, sampler=infinite_sampler, batch_size=args.batch_size, num_workers=4,
                              prefetch_factor=2)
    test_loader = DataLoader(test_set, batch_size=args.test_batch_size, shuffle=False, num_workers=4)

    """ Initialize model and optimizer """
    model = LatentModulatedSIREN(
        in_size=args.in_size,
        out_size=args.out_size,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        latent_modulation_dim=args.latent_modulation_dim,
        w0=args.w0,
        w0_increments=args.w0_increment,
        modulate_shift=args.modulate_shift,
        modulate_scale=args.modulate_scale,
        enable_skip_connections=args.enable_skip_connections,
    ).to(device)

    model.modulations = torch.zeros(size=[args.batch_size, args.latent_modulation_dim], requires_grad=True).to(device)
    model = ModelWrapper(args, model)

    meta_optimizer = optim.AdamW(params=model.parameters(), lr=args.meta_lr)

    """ Define training and test functions """
    train_function = train_step
    test_function = test_model

    """ Define logger """
    fname = (f'{args.dataset}_bs{args.batch_size}_inner{args.inner_steps}_size{args.img_size}_gamma{args.data_ratio}_'
             f'{args.config.split("/")[-1].split(".yaml")[0]}')
    logger = Logger(fname, ask=args.resume_path is None, rank=args.gpu_id)
    logger.log(args)
    logger.log(model)

    """ Perform training """
    trainer(args, train_function, test_function, model, meta_optimizer, train_loader, test_loader, logger)

    """ Close logger """
    logger.close_writer()


if __name__ == "__main__":
    args = parse_args()
    main(args)
