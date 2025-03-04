import torch
from torch.utils.data import DataLoader

from common.args import parse_args
from common.utils import set_random_seed, load_model
from data.dataset import get_dataset
from eval.maml_full_eval import test_model
from models.inrs import LatentModulatedSIREN
from models.model_wrapper import ModelWrapper


def main(args):
    """
    Main function to call for running an evaluation procedure.
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
    torch.backends.cudnn.benchmark = False

    """ Define test dataset """
    test_set = get_dataset(args, only_test=True)
    test_loader = DataLoader(test_set, batch_size=args.test_batch_size, shuffle=False, num_workers=4, pin_memory=True,
                             drop_last=True)

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
    model.modulations = torch.zeros(size=[args.test_batch_size, args.latent_modulation_dim], requires_grad=True).to(device)
    model = ModelWrapper(args, model)
    load_model(args, model)

    """ Define test function """
    test_model(args, model, test_loader, logger=None)


if __name__ == "__main__":
    args = parse_args()
    main(args)
