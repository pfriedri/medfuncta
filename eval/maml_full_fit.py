import torch
import torchvision.transforms as transforms

from tqdm import tqdm
from train.maml_boot import inner_adapt_test_scale

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def fit_nfs(args, model_wrapper, dataloader, set=None):

    model_wrapper.model.eval()
    model_wrapper.coord_init()

    for n, data in enumerate(tqdm(dataloader)):
        data, label = data
        data = data.to(device)
        batch_size = data.size(0)
        model_wrapper.model.reset_modulations()

        _ = inner_adapt_test_scale(model_wrapper=model_wrapper, data=data, step_size=args.inner_lr,
                                   num_steps=args.inner_steps_test, first_order=True,
                                   sample_type=args.sample_type, scale_type='grad')

        if set == 'test':
            with torch.no_grad():
                pred = model_wrapper().clamp(0, 1)
            if n < 100:
                # Convert to PIL image
                to_pil = transforms.ToPILImage()
                image = to_pil(data.squeeze())
                image.save(f"./imgs/{n}_input.png")
                image = to_pil(pred.squeeze())
                image.save(f"./imgs/{n}_recon.png")

        for i in range(batch_size):
            datapoint = {
                'modulations': model_wrapper.model.modulations[i].detach().cpu(),
                'label': label[i].detach().cpu()
            }
            sdir = args.save_dir + f'/{set}/' + f'datapoint_{(n * batch_size) + i}.pt'
            torch.save(datapoint, sdir)
    return
