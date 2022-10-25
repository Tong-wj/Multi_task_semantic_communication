#
# Authors: Simon Vandenhende
# Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)

from evaluation.evaluate_utils import PerformanceMeter
from utils.utils import AverageMeter, ProgressMeter, get_output


def get_loss_meters(cfg):
    """ Return dictionary with loss meters to monitor training """
    all_tasks = cfg.ALL_TASKS.NAMES
    tasks = cfg.TASKS.NAMES


    if cfg['model'] == 'mti_net': # Extra losses at multiple scales
        losses = {}
        for scale in range(4):
            for task in all_tasks:
                losses['scale_%d_%s' %(scale, task)] = AverageMeter('Loss scale-%d %s ' %(scale+1, task), ':.4e')
        for task in tasks:
            losses[task] = AverageMeter('Loss %s' %(task), ':.4e')


    elif cfg['model'] == 'pad_net': # Extra losses because of deepsupervision
        losses = {}
        for task in all_tasks:
            losses['deepsup_%s' %(task)] = AverageMeter('Loss deepsup %s' %(task), ':.4e')
        for task in tasks:
            losses[task] = AverageMeter('Loss %s' %(task), ':.4e')


    else: # Only losses on the main task.
        losses = {task: AverageMeter('Loss %s' %(task), ':.4e') for task in tasks}

    losses['dynamic_jscc'] = AverageMeter('Loss dynamic_jscc', ':.4e')
    losses['cpp'] = AverageMeter('CPP', ':.4e')
    losses['total'] = AverageMeter('Loss Total', ':.4e')
    return losses


def train_vanilla(cfg, train_loader, model, criterion, optimizer, epoch):
    """ Vanilla training with fixed loss weights """
    losses = get_loss_meters(cfg)
    performance_meter = PerformanceMeter(cfg)
    progress = ProgressMeter(len(train_loader),
        [v for v in losses.values()], prefix="Epoch: [{}]".format(epoch))

    model.train()
    
    for i, batch in enumerate(train_loader):
        # Forward pass
        images = batch['image'].cuda(non_blocking=True)
        targets = {task: batch[task].cuda(non_blocking=True) for task in cfg.ALL_TASKS.NAMES}
        output = model(images)
        
        # import pdb
        # pdb.set_trace()
        # Measure loss and performance
        loss_dict = criterion(output, targets)
        loss_dict['dynamic_jscc'] = output['dynamic_jscc_loss']
        loss_dict['cpp'] = output['cpp']
        loss_dict['total'] += output['dynamic_jscc_loss']
        for k, v in loss_dict.items():
            losses[k].update(v.item())
        performance_meter.update({t: get_output(output[t], t) for t in cfg.TASKS.NAMES}, 
                                 {t: targets[t] for t in cfg.TASKS.NAMES})
        
        # Backward
        optimizer.zero_grad()
        loss_dict['total'].backward()
        optimizer.step()

        if i % 25 == 0:
            progress.display(i)

    eval_results = performance_meter.get_score(verbose = True)

    return eval_results
