import os.path

import torch
from torch.utils.data import DataLoader

from common.args import parse_args
from common.utils import set_random_seed, load_model
from data.dataset import get_dataset
from eval.maml_full_fit import fit_nfs
from models.inrs import LatentModulatedSIREN
from models.model_wrapper import ModelWrapper


def main(args):
    """
    Main function to call for fitting neural fields to a whole dataset (having pretrained shared weights).
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

    """ Define dataset that you want to convert to NFs """
    train, val, test = get_dataset(args, all=True)
    train_loader = DataLoader(train, batch_size=args.test_batch_size, shuffle=False, num_workers=4, pin_memory=True,
                              drop_last=True)
    val_loader = DataLoader(val, batch_size=args.test_batch_size, shuffle=False, num_workers=4, pin_memory=True,
                            drop_last=True)
    test_loader = DataLoader(test, batch_size=args.test_batch_size, shuffle=False, num_workers=4, pin_memory=True,
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

    if not os.path.exists(args.save_dir):
        print(f'Create: {args.save_dir}')
        os.mkdir(args.save_dir)

    """ Create training set """
    if not os.path.exists(args.save_dir + 'train'):
        print(f'Create: {args.save_dir}'+'train/')
        os.mkdir(args.save_dir + 'train/')
    fit_nfs(args, model, train_loader, set='train')
    print("Created MedFuncta Set: Training")

    """ Create validation set """
    if not os.path.exists(args.save_dir + 'val'):
        print(f'Create: {args.save_dir}' + 'val/')
        os.mkdir(args.save_dir + 'val/')
    fit_nfs(args, model, val_loader, set='val')
    print("Created MedFuncta Set: Validation")

    """ Create test set """
    if not os.path.exists(args.save_dir + 'test'):
        print(f'Create: {args.save_dir}' + 'test/')
        os.mkdir(args.save_dir + 'test')
    fit_nfs(args, model, test_loader, set='test')
    print("Created MedFuncta Set: Test")

    """ Save the model to save_dir folder """
    model_path = args.save_dir + 'model.pt'
    torch.save(model.model, model_path)
    print("DONE")


if __name__ == "__main__":
    args = parse_args()
    main(args)
