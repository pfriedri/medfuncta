import torch
import lpips
import torch.nn.functional as F
from tqdm import tqdm
from pytorch_msssim import ssim

from common.utils import MetricLogger, psnr
from train.maml_boot import inner_adapt_test_scale

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def test_model(args, model_wrapper, test_loader, logger=None):
    metric_logger = MetricLogger(delimiter="  ")

    if logger is None:
        log_ = print
    else:
        log_ = logger.log

    model_wrapper.model.eval()
    model_wrapper.coord_init()

    lpips_score = lpips.LPIPS(net='alex').to(device)

    for n, data in enumerate(tqdm(test_loader)):
        data, _ = data
        data = data.to(device)
        batch_size = data.size(0)
        model_wrapper.model.reset_modulations()

        _ = inner_adapt_test_scale(model_wrapper=model_wrapper, data=data, step_size=args.inner_lr,
                                   num_steps=args.inner_steps_test, first_order=True,
                                   sample_type=args.sample_type, scale_type='grad')

        with torch.no_grad():
            pred = model_wrapper().clamp(0, 1)

        if args.data_type == 'img':
            lpips_results = lpips_score((pred * 2 - 1), (data * 2 - 1)).mean()
            mse_results = F.mse_loss(data.view(batch_size, -1), pred.reshape(batch_size, -1), reduce=False).mean()
            psnr_results = psnr(
                F.mse_loss(data.view(batch_size, -1), pred.reshape(batch_size, -1), reduce=False).mean(dim=1)
            ).mean()
            ssim_results = ssim(pred, data, data_range=1.).mean()

        elif args.data_type == 'img3d':
            mse_results = F.mse_loss(data.view(batch_size, -1), pred.reshape(batch_size, -1), reduce=False).mean()
            psnr_results = psnr(
                F.mse_loss(data.view(batch_size, -1), pred.reshape(batch_size, -1), reduce=False).mean(dim=1)
            ).mean()
            ssim_results = ssim(pred, data, data_range=1.).mean()
            lpips_results = torch.zeros_like(psnr_results)

        elif args.data_type == 'timeseries':
            mse_results = F.mse_loss(data.view(batch_size, -1), pred.reshape(batch_size, -1), reduce=False).mean()
            psnr_results = psnr(
                F.mse_loss(data.view(batch_size, -1), pred.reshape(batch_size, -1), reduce=False).mean(dim=1)
            ).mean()
            ssim_results = torch.zeros_like(psnr_results)
            lpips_results = torch.zeros_like(psnr_results)

        else:
            raise NotImplementedError()

        metric_logger.meters['lpips'].update(lpips_results.item(), n=batch_size)
        metric_logger.meters['psnr'].update(psnr_results.item(), n=batch_size)
        metric_logger.meters['mse'].update(mse_results.item(), n=batch_size)
        metric_logger.meters['ssim'].update(ssim_results.item(), n=batch_size)

        if n % 10 == 0:
            # gather the stats from all processes
            metric_logger.synchronize_between_processes()

            log_(f'*[EVAL {n}][PSNR %.6f][LPIPS %.6f][SSIM %.6f][MSE %.6f]' %
                 (metric_logger.psnr.global_avg, metric_logger.lpips.global_avg,
                  metric_logger.ssim.global_avg, metric_logger.mse.global_avg))

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    log_(f'*[EVAL Final][PSNR %.8f][LPIPS %.8f][SSIM %.8f][MSE %.8f]' %
         (metric_logger.psnr.global_avg, metric_logger.lpips.global_avg,
          metric_logger.ssim.global_avg, metric_logger.mse.global_avg))

    return
