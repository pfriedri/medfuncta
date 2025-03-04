import torch
from common.utils import psnr


def modulation_consistency(modulations, modulations_bootstrapped, bs):
    """
    A function that calculates the L2-distance between the modulations and a bootstrapped target.
    Proposed in 'Learning Large-scale Neural Fields via Context Pruned Meta-Learning' by Jihoon Tack, et al. (2023)

    Everything is implemented to use this bootstrap correction. It is however NOT USED IN OUR PAPER.
    """
    updated_modulation = modulations_bootstrapped - modulations
    updated_modulation = updated_modulation.view(bs, -1)
    modulation_norm = torch.mean(updated_modulation ** 2, dim=-1)
    return modulation_norm


def get_grad_norm(grads, detach=True):
    grad_norm_list = []
    for grad in grads:
        if grad is None:
            grad_norm = 0
        else:
            if detach:
                grad_norm = torch.norm(grad.data, p=2, keepdim=True).unsqueeze(dim=0)
            else:
                grad_norm = torch.norm(grad, p=2, keepdim=True).unsqueeze(dim=0)

        grad_norm_list.append(grad_norm)
    return torch.norm(torch.cat(grad_norm_list, dim=0), p=2, dim=1)


def train_step(args, step, model_wrapper, optimizer, data, metric_logger, logger):
    """
    Function that performs a single meta update
    """
    model_wrapper.model.train()
    model_wrapper.coord_init()  # Reset coordinates
    model_wrapper.model.reset_modulations()  # Reset modulations (zero-initialization)

    batch_size = data.size(0)

    if step % args.print_step == 0:
        learned_init = model_wrapper()
        input = data

    """ Inner-loop optimization for G steps """
    loss_in = inner_adapt(model_wrapper=model_wrapper, data=data, step_size=args.inner_lr,
                          num_steps=args.inner_steps, first_order=False, sample_type=args.sample_type)

    """ Compute reconstruction loss using full context set"""
    model_wrapper.coord_init()
    modulations = model_wrapper.model.modulations.clone()  # Store modulations for consistency loss (not used)
    loss_out = model_wrapper(data)  # Compute reconstruction loss
    if step % args.print_step == 0:
        images = model_wrapper()  # Sample images

    """ Bootstrap correction for additional steps (NOT USED IN THIS PAPER) """
    _ = inner_adapt(model_wrapper=model_wrapper, data=data, step_size=args.inner_lr_boot,
                    num_steps=args.inner_steps_boot, first_order=True)
    modulations_bootstrapped = model_wrapper.model.modulations.detach()
    if step % args.print_step == 0:
        target_boot = model_wrapper()

    """ Modulation consistency loss and loss aggregation (WE ONLY USE RECONSTRUCTION LOSS) """
    loss_boot = modulation_consistency(modulations, modulations_bootstrapped, bs=batch_size)
    loss_boot_weighted = args.lam * loss_boot
    loss = loss_out.mean() + loss_boot_weighted.mean()

    """ Meta update (optimize shared weights) """
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model_wrapper.model.parameters(), 1.0)
    optimizer.step()
    torch.cuda.synchronize()

    """ Track stats"""
    metric_logger.meters['loss_inner'].update(loss_in.mean().item(), n=batch_size)
    metric_logger.meters['loss_outer'].update(loss_out.mean().item(), n=batch_size)
    metric_logger.meters['psnr_inner'].update(psnr(loss_in).mean().item(), n=batch_size)
    metric_logger.meters['psnr_outer'].update(psnr(loss_out).mean().item(), n=batch_size)
    metric_logger.meters['loss_boot'].update(loss_boot_weighted.mean().item(), n=batch_size)
    metric_logger.synchronize_between_processes()

    if step % args.print_step == 0:
        logger.scalar_summary('train/loss_inner', metric_logger.loss_inner.global_avg, step)
        logger.scalar_summary('train/loss_outer', metric_logger.loss_outer.global_avg, step)
        logger.scalar_summary('train/psnr_inner', metric_logger.psnr_inner.global_avg, step)
        logger.scalar_summary('train/psnr_outer', metric_logger.psnr_outer.global_avg, step)
        logger.scalar_summary('train/loss_boot', metric_logger.loss_boot.global_avg, step)
        logger.log_image('train/img_in', input, step)
        logger.log_image('train/learninit', learned_init, step)
        logger.log_image('train/img_inner', images, step)
        logger.log_image('train/img_bst', target_boot, step)

        logger.log('[TRAIN] [Step %3d] [LossInner %f] [LossOuter %f] [PSNRInner %.3f] [PSNROuter %.3f]' %
                   (step, metric_logger.loss_inner.global_avg, metric_logger.loss_outer.global_avg,
                    metric_logger.psnr_inner.global_avg, metric_logger.psnr_outer.global_avg))

    metric_logger.reset()


def inner_adapt(model_wrapper, data, step_size=1e-2, num_steps=3, first_order=False, sample_type='none'):
    loss = 0.  # Initialize outer_loop loss

    """ Perform num_step (G) inner-loop updates """
    for step_inner in range(num_steps):
        if sample_type != 'none':
            model_wrapper.sample_coordinates(sample_type, data)  # Sample coordinates for the training step
        loss = inner_loop_step(model_wrapper, data, step_size, first_order)

    return loss


def inner_loop_step(model_wrapper, data, inner_lr=1e-2, first_order=False):
    batch_size = data.size(0)

    with torch.enable_grad():
        loss = model_wrapper(data)
        grads = torch.autograd.grad(
            loss.mean() * batch_size,
            model_wrapper.model.modulations,
            create_graph=not first_order,
        )[0]
        model_wrapper.model.modulations = model_wrapper.model.modulations - inner_lr * grads
    return loss


def inner_adapt_test_scale(model_wrapper, data, step_size=1e-2, num_steps=3, first_order=False, sample_type='none',
                           scale_type='grad'):
    loss = 0.  # Initialize outer_loop loss

    for step_inner in range(num_steps):
        if sample_type != 'none':
            model_wrapper.sample_coordinates(sample_type, data)

        loss = inner_loop_step_tt_gradscale(model_wrapper, data, step_size, first_order, scale_type)

    return loss


def inner_loop_step_tt_gradscale(model_wrapper, data, inner_lr=1e-2, first_order=False, scale_type='grad'):
    batch_size = data.size(0)
    model_wrapper.model.zero_grad()

    with torch.enable_grad():
        subsample_loss = model_wrapper(data)
        subsample_grad = torch.autograd.grad(
            subsample_loss.mean() * batch_size,
            model_wrapper.model.modulations,
            create_graph=False,
            allow_unused=True
        )[0]

    model_wrapper.model.zero_grad()
    model_wrapper.coord_init()

    with torch.enable_grad():
        loss = model_wrapper(data)

        grads = torch.autograd.grad(
            loss.mean() * batch_size,
            model_wrapper.model.modulations,
            create_graph=not first_order,
            allow_unused=True
        )[0]

    if scale_type == 'grad':
        # Gradient rescaling at test-time
        subsample_grad_norm = get_grad_norm(subsample_grad, detach=True)
        grad_norm = get_grad_norm(grads, detach=True)
        grad_scale = subsample_grad_norm / (grad_norm + 1e-16)
        grad_scale_ = grad_scale.view((batch_size,) + (1,) * (len(grads.shape) - 1)).detach()
    else:
        raise NotImplementedError()

    model_wrapper.model.modulations = model_wrapper.model.modulations - inner_lr * grad_scale_ * grads

    return loss
