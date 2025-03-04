import torch

from common.utils import MetricLogger, psnr
from train.maml_boot import inner_adapt_test_scale

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def test_model(args, step, model_wrapper, test_loader, logger=None):
    metric_logger = MetricLogger(delimiter="  ")

    if logger is None:
        log_ = print
    else:
        log_ = logger.log

    model_wrapper.model.eval()
    model_wrapper.coord_init()

    for n, data in enumerate(test_loader):
        if n * args.test_batch_size > args.num_test_signals:
            break

        data, _ = data
        data = data.float().to(device)
        batch_size = data.size(0)
        model_wrapper.model.reset_modulations()

        if n == 0:
            input = data

        loss_in_tt_gradscale = inner_adapt_test_scale(model_wrapper=model_wrapper, data=data, step_size=args.inner_lr,
                                                      num_steps=args.inner_steps_test, first_order=True,
                                                      sample_type=args.sample_type, scale_type='grad')

        psnr_in_tt_gradscale = psnr(loss_in_tt_gradscale)

        """ Outer loss aggregation """
        with torch.no_grad():
            loss_out_tt_gradscale = model_wrapper(data)
            psnr_out_tt_gradscale = psnr(loss_out_tt_gradscale)
            if n == 0:
                out = model_wrapper()

        metric_logger.meters['loss_inner_tt_gradscale'].update(loss_in_tt_gradscale.mean().item(), n=batch_size)
        metric_logger.meters['loss_outer_tt_gradscale'].update(loss_out_tt_gradscale.mean().item(), n=batch_size)
        metric_logger.meters['psnr_inner_tt_gradscale'].update(psnr_in_tt_gradscale.mean().item(), n=batch_size)
        metric_logger.meters['psnr_outer_tt_gradscale'].update(psnr_out_tt_gradscale.mean().item(), n=batch_size)

    metric_logger.synchronize_between_processes()
    log_('*[EVAL Gradscale TestTime][LossInnerGSTT %.3f][LossOuterGSTT %.3f][PSNRInnerGSTT %.3f][PSNROuterGSTT %.3f]' %
         (metric_logger.loss_inner_tt_gradscale.global_avg, metric_logger.loss_outer_tt_gradscale.global_avg,
          metric_logger.psnr_inner_tt_gradscale.global_avg, metric_logger.psnr_outer_tt_gradscale.global_avg))

    if logger is not None:
        logger.scalar_summary('eval/loss_inner_TT_gradscale', metric_logger.loss_inner_tt_gradscale.global_avg, step)
        logger.scalar_summary('eval/loss_outer_TT_gradscale', metric_logger.loss_outer_tt_gradscale.global_avg, step)
        logger.scalar_summary('eval/psnr_inner_TT_gradscale', metric_logger.psnr_inner_tt_gradscale.global_avg, step)
        logger.scalar_summary('eval/psnr_outer_TT_gradscale', metric_logger.psnr_outer_tt_gradscale.global_avg, step)
        logger.log_image('eval/img_in', input, step)
        logger.log_image('eval/img_adapt_tt', out, step)
    return metric_logger.psnr_outer_tt_gradscale.global_avg
