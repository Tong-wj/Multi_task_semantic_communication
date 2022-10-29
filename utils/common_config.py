#
# Authors: Simon Vandenhende
# Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)

import os
import copy
import torch
import torch.nn.functional as F

from torchvision import transforms
from torch.utils.data import DataLoader
from utils.custom_collate import collate_mil


"""
    Model getters 
"""
def get_backbone(cfg):
    """ Return the backbone """

    if cfg['backbone'] == 'resnet18':
        from models.resnet import resnet18
        backbone = resnet18(cfg['backbone_kwargs']['pretrained'])
        backbone_channels = 512
    
    elif cfg['backbone'] == 'resnet50':
        from models.resnet import resnet50
        backbone = resnet50(cfg['backbone_kwargs']['pretrained'])
        backbone_channels = 2048

    elif cfg['backbone'] == 'hrnet_w18':
        from models.seg_hrnet import hrnet_w18
        backbone = hrnet_w18(cfg['backbone_kwargs']['pretrained'])
        backbone_channels = [18, 36, 72, 144]
    
    else:
        raise NotImplementedError

    if cfg['backbone_kwargs']['dilated']: # Add dilated convolutions
        assert(cfg['backbone'] in ['resnet18', 'resnet50'])
        from models.resnet_dilated import ResnetDilated
        backbone = ResnetDilated(backbone)

    if 'fuse_hrnet' in cfg['backbone_kwargs'] and cfg['backbone_kwargs']['fuse_hrnet']: # Fuse the multi-scale HRNet features
        from models.seg_hrnet import HighResolutionFuse
        backbone = torch.nn.Sequential(backbone, HighResolutionFuse(backbone_channels, 256))
        backbone_channels = sum(backbone_channels)

    return backbone, backbone_channels

def get_dynamic_jscc(cfg, backbone_channels):
    """ Return the dynamic_jscc """

    from models.dynamic_jscc import DynaJSCC
    return DynaJSCC(cfg, backbone_channels)

def get_head(cfg, backbone_channels, task):
    """ Return the decoder head """

    if cfg['head'] == 'deeplab':
        from models.aspp import DeepLabHead
        return DeepLabHead(backbone_channels, cfg.TASKS.NUM_OUTPUT[task])

    elif cfg['head'] == 'hrnet':
        from models.seg_hrnet import HighResolutionHead
        return HighResolutionHead(backbone_channels, cfg.TASKS.NUM_OUTPUT[task])

    else:
        raise NotImplementedError


def get_model(cfg):
    """ Return the model """


    backbone, backbone_channels = get_backbone(cfg)
    dynamic_jscc = get_dynamic_jscc(cfg['dynamic_jscc'], backbone_channels)

    if cfg['setup'] == 'single_task':
        from models.models import SingleTaskModel
        task = cfg.TASKS.NAMES[0]
        head = get_head(cfg, backbone_channels, task)
        model = SingleTaskModel(backbone, dynamic_jscc, head, task)


    elif cfg['setup'] == 'multi_task':
        if cfg['model'] == 'baseline':
            from models.models import MultiTaskModel
            heads = torch.nn.ModuleDict({task: get_head(cfg, backbone_channels, task) for task in cfg.TASKS.NAMES})
            model = MultiTaskModel(backbone, dynamic_jscc, heads, cfg.TASKS.NAMES)


        elif cfg['model'] == 'cross_stitch':
            from models.models import SingleTaskModel
            from models.cross_stitch import CrossStitchNetwork
            
            # Load single-task models
            backbone_dict, decoder_dict = {}, {}
            for task in cfg.TASKS.NAMES:
                model = SingleTaskModel(copy.deepcopy(backbone), get_head(cfg, backbone_channels, task), task)
                model = torch.nn.DataParallel(model)
                model.load_state_dict(torch.load(os.path.join(cfg['root_dir'], cfg['train_db_name'], cfg['backbone'], 'single_task', task, 'best_model.pth.tar')))
                backbone_dict[task] = model.module.backbone
                decoder_dict[task] = model.module.decoder
            
            # Stitch the single-task models together
            model = CrossStitchNetwork(cfg, torch.nn.ModuleDict(backbone_dict), torch.nn.ModuleDict(decoder_dict), 
                                        **cfg['model_kwargs']['cross_stitch_kwargs'])


        elif cfg['model'] == 'nddr_cnn':
            from models.models import SingleTaskModel
            from models.nddr_cnn import NDDRCNN
            
            # Load single-task models
            backbone_dict, decoder_dict = {}, {}
            for task in cfg.TASKS.NAMES:
                model = SingleTaskModel(copy.deepcopy(backbone), get_head(cfg, backbone_channels, task), task)
                model = torch.nn.DataParallel(model)
                model.load_state_dict(torch.load(os.path.join(cfg['root_dir'], cfg['train_db_name'], cfg['backbone'], 'single_task', task, 'best_model.pth.tar')))
                backbone_dict[task] = model.module.backbone
                decoder_dict[task] = model.module.decoder
            
            # Stitch the single-task models together
            model = NDDRCNN(cfg, torch.nn.ModuleDict(backbone_dict), torch.nn.ModuleDict(decoder_dict), 
                                        **cfg['model_kwargs']['nddr_cnn_kwargs'])


        elif cfg['model'] == 'mtan':
            from models.mtan import MTAN
            heads = torch.nn.ModuleDict({task: get_head(cfg, backbone_channels, task) for task in cfg.TASKS.NAMES})
            model = MTAN(cfg, backbone, heads, **cfg['model_kwargs']['mtan_kwargs'])


        elif cfg['model'] == 'pad_net':
            from models.padnet import PADNet
            model = PADNet(cfg, backbone, backbone_channels)
        

        elif cfg['model'] == 'mti_net':
            from models.mti_net import MTINet
            heads = torch.nn.ModuleDict({task: get_head(cfg, backbone_channels, task) for task in cfg.TASKS.NAMES})
            model = MTINet(cfg, backbone, backbone_channels, heads)


        else:
            raise NotImplementedError('Unknown model {}'.format(cfg['model']))


    else:
        raise NotImplementedError('Unknown setup {}'.format(cfg['setup']))
    

    return model


"""
    Transformations, datasets and dataloaders
"""
def get_transformations(cfg):
    """ Return transformations for training and evaluationg """
    from data import custom_transforms as tr

    # Training transformations
    if cfg['train_db_name'] == 'NYUD':
        # Horizontal flips with probability of 0.5
        transforms_tr = [tr.RandomHorizontalFlip()]
        
        # Rotations and scaling
        transforms_tr.extend([tr.ScaleNRotate(rots=[0], scales=[1.0, 1.2, 1.5],
                                              flagvals={x: cfg.ALL_TASKS.FLAGVALS[x] for x in cfg.ALL_TASKS.FLAGVALS})])

    elif cfg['train_db_name'] == 'PASCALContext':
        # Horizontal flips with probability of 0.5
        transforms_tr = [tr.RandomHorizontalFlip()]
        
        # Rotations and scaling
        transforms_tr.extend([tr.ScaleNRotate(rots=(-20, 20), scales=(.75, 1.25),
                                              flagvals={x: cfg.ALL_TASKS.FLAGVALS[x] for x in cfg.ALL_TASKS.FLAGVALS})])

    else:
        raise ValueError('Invalid train db name'.format(cfg['train_db_name']))


    # Fixed Resize to input resolution
    transforms_tr.extend([tr.FixedResize(resolutions={x: tuple(cfg.TRAIN.SCALE) for x in cfg.ALL_TASKS.FLAGVALS},
                                         flagvals={x: cfg.ALL_TASKS.FLAGVALS[x] for x in cfg.ALL_TASKS.FLAGVALS})])
    transforms_tr.extend([tr.AddIgnoreRegions(), tr.ToTensor(),
                          tr.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    transforms_tr = transforms.Compose(transforms_tr)

    
    # Testing (during training transforms)
    transforms_ts = []
    transforms_ts.extend([tr.FixedResize(resolutions={x: tuple(cfg.TEST.SCALE) for x in cfg.TASKS.FLAGVALS},
                                         flagvals={x: cfg.TASKS.FLAGVALS[x] for x in cfg.TASKS.FLAGVALS})])
    transforms_ts.extend([tr.AddIgnoreRegions(), tr.ToTensor(),
                          tr.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    transforms_ts = transforms.Compose(transforms_ts)

    return transforms_tr, transforms_ts


def get_train_dataset(cfg, transforms):
    """ Return the train dataset """

    db_name = cfg['train_db_name']
    print('Preparing train loader for db: {}'.format(db_name))

    if db_name == 'PASCALContext':
        from data.pascal_context import PASCALContext
        database = PASCALContext(split=['train'], transform=transforms, retname=True,
                                          do_semseg='semseg' in cfg.ALL_TASKS.NAMES,
                                          do_edge='edge' in cfg.ALL_TASKS.NAMES,
                                          do_normals='normals' in cfg.ALL_TASKS.NAMES,
                                          do_sal='sal' in cfg.ALL_TASKS.NAMES,
                                          do_human_parts='human_parts' in cfg.ALL_TASKS.NAMES,
                                          overfit=cfg['overfit'])

    elif db_name == 'NYUD':
        from data.nyud import NYUD_MT
        database = NYUD_MT(split='train', transform=transforms, do_edge='edge' in cfg.ALL_TASKS.NAMES, 
                                    do_semseg='semseg' in cfg.ALL_TASKS.NAMES, 
                                    do_normals='normals' in cfg.ALL_TASKS.NAMES, 
                                    do_depth='depth' in cfg.ALL_TASKS.NAMES, overfit=cfg['overfit'])

    else:
        raise NotImplemented("train_db_name: Choose among PASCALContext and NYUD")

    return database


def get_train_dataloader(cfg, dataset):
    """ Return the train dataloader """

    trainloader = DataLoader(dataset, batch_size=cfg['trBatch'], shuffle=True, drop_last=True,
                             num_workers=cfg['nworkers'], collate_fn=collate_mil)
    return trainloader


def get_val_dataset(cfg, transforms):
    """ Return the validation dataset """

    db_name = cfg['val_db_name']
    print('Preparing val loader for db: {}'.format(db_name))

    if db_name == 'PASCALContext':
        from data.pascal_context import PASCALContext
        database = PASCALContext(split=['val'], transform=transforms, retname=True,
                                      do_semseg='semseg' in cfg.TASKS.NAMES,
                                      do_edge='edge' in cfg.TASKS.NAMES,
                                      do_normals='normals' in cfg.TASKS.NAMES,
                                      do_sal='sal' in cfg.TASKS.NAMES,
                                      do_human_parts='human_parts' in cfg.TASKS.NAMES,
                                    overfit=cfg['overfit'])
    
    elif db_name == 'NYUD':
        from data.nyud import NYUD_MT
        database = NYUD_MT(split='val', transform=transforms, do_edge='edge' in cfg.TASKS.NAMES, 
                                do_semseg='semseg' in cfg.TASKS.NAMES, 
                                do_normals='normals' in cfg.TASKS.NAMES, 
                                do_depth='depth' in cfg.TASKS.NAMES, overfit=cfg['overfit'])

    else:
        raise NotImplemented("test_db_name: Choose among PASCALContext and NYUD")

    return database


def get_val_dataloader(cfg, dataset):
    """ Return the validation dataloader """

    testloader = DataLoader(dataset, batch_size=cfg['valBatch'], shuffle=False, drop_last=False,
                            num_workers=cfg['nworkers'])
    return testloader


""" 
    Loss functions 
"""
def get_loss(cfg, task=None):
    """ Return loss function for a specific task """

    if task == 'edge':
        from losses.loss_functions import BalancedCrossEntropyLoss
        criterion = BalancedCrossEntropyLoss(size_average=True, pos_weight=cfg['edge_w'])

    elif task == 'semseg' or task == 'human_parts':
        from losses.loss_functions import SoftMaxwithLoss
        criterion = SoftMaxwithLoss()

    elif task == 'normals':
        from losses.loss_functions import NormalsLoss
        criterion = NormalsLoss(normalize=True, size_average=True, norm=cfg['normloss'])

    elif task == 'sal':
        from losses.loss_functions import BalancedCrossEntropyLoss
        criterion = BalancedCrossEntropyLoss(size_average=True)

    elif task == 'depth':
        from losses.loss_functions import DepthLoss
        criterion = DepthLoss(cfg['depthloss'])

    else:
        raise NotImplementedError('Undefined Loss: Choose a task among '
                                  'edge, semseg, human_parts, sal, depth, or normals')

    return criterion


def get_criterion(cfg):
    """ Return training criterion for a given setup """

    if cfg['setup'] == 'single_task':
        from losses.loss_schemes import SingleTaskLoss
        task = cfg.TASKS.NAMES[0]
        loss_ft = get_loss(cfg, task)
        return SingleTaskLoss(loss_ft, task)

    
    elif cfg['setup'] == 'multi_task':
        if cfg['loss_kwargs']['loss_scheme'] == 'baseline': # Fixed weights
            from losses.loss_schemes import MultiTaskLoss
            loss_ft = torch.nn.ModuleDict({task: get_loss(cfg, task) for task in cfg.TASKS.NAMES})
            loss_weights = cfg['loss_kwargs']['loss_weights']
            return MultiTaskLoss(cfg.TASKS.NAMES, loss_ft, loss_weights)


        elif cfg['loss_kwargs']['loss_scheme'] == 'pad_net': # Fixed weights but w/ deep supervision
            from losses.loss_schemes import PADNetLoss
            loss_ft = torch.nn.ModuleDict({task: get_loss(cfg, task) for task in cfg.ALL_TASKS.NAMES})
            loss_weights = cfg['loss_kwargs']['loss_weights']
            return PADNetLoss(cfg.TASKS.NAMES, cfg.AUXILARY_TASKS.NAMES, loss_ft, loss_weights)
 

        elif cfg['loss_kwargs']['loss_scheme'] == 'mti_net': # Fixed weights but at multiple scales
            from losses.loss_schemes import MTINetLoss
            loss_ft = torch.nn.ModuleDict({task: get_loss(cfg, task) for task in set(cfg.ALL_TASKS.NAMES)})
            loss_weights = cfg['loss_kwargs']['loss_weights']
            return MTINetLoss(cfg.TASKS.NAMES, cfg.AUXILARY_TASKS.NAMES, loss_ft, loss_weights)

        
        else:
            raise NotImplementedError('Unknown loss scheme {}'.format(cfg['loss_kwargs']['loss_scheme']))

    else:
        raise NotImplementedError('Unknown setup {}'.format(cfg['setup']))


"""
    Optimizers and schedulers
"""
def get_optimizer(cfg, model):
    """ Return optimizer for a given model and setup """

    if cfg['model'] == 'cross_stitch': # Custom learning rate for cross-stitch
        print('Optimizer uses custom scheme for cross-stitch nets')
        cross_stitch_params = [param for name, param in model.named_parameters() if 'cross_stitch' in name]
        single_task_params = [param for name, param in model.named_parameters() if not 'cross_stitch' in name]
        assert(cfg['optimizer'] == 'sgd') # Adam seems to fail for cross-stitch nets
        optimizer = torch.optim.SGD([{'params': cross_stitch_params, 'lr': 100*cfg['optimizer_kwargs']['lr']},
                                     {'params': single_task_params, 'lr': cfg['optimizer_kwargs']['lr']}],
                                        momentum = cfg['optimizer_kwargs']['momentum'], 
                                        nesterov = cfg['optimizer_kwargs']['nesterov'],
                                        weight_decay = cfg['optimizer_kwargs']['weight_decay'])


    elif cfg['model'] == 'nddr_cnn': # Custom learning rate for nddr-cnn
        print('Optimizer uses custom scheme for nddr-cnn nets')
        nddr_params = [param for name, param in model.named_parameters() if 'nddr' in name]
        single_task_params = [param for name, param in model.named_parameters() if not 'nddr' in name]
        assert(cfg['optimizer'] == 'sgd') # Adam seems to fail for nddr-cnns 
        optimizer = torch.optim.SGD([{'params': nddr_params, 'lr': 100*cfg['optimizer_kwargs']['lr']},
                                     {'params': single_task_params, 'lr': cfg['optimizer_kwargs']['lr']}],
                                        momentum = cfg['optimizer_kwargs']['momentum'], 
                                        nesterov = cfg['optimizer_kwargs']['nesterov'],
                                        weight_decay = cfg['optimizer_kwargs']['weight_decay'])


    else: # Default. Same larning rate for all params
        print('Optimizer uses a single parameter group - (Default)')
        params = model.parameters()
    
        if cfg['optimizer'] == 'sgd':
            optimizer = torch.optim.SGD(params, **cfg['optimizer_kwargs'])

        elif cfg['optimizer'] == 'adam':
            optimizer = torch.optim.Adam(params, **cfg['optimizer_kwargs'])
        
        else:
            raise ValueError('Invalid optimizer {}'.format(cfg['optimizer']))

    return optimizer
   

def adjust_learning_rate(cfg, optimizer, epoch):
    """ Adjust the learning rate """

    lr = cfg['optimizer_kwargs']['lr']
    
    if cfg['scheduler'] == 'step':
        steps = np.sum(epoch > np.array(cfg['scheduler_kwargs']['lr_decay_epochs']))
        if steps > 0:
            lr = lr * (cfg['scheduler_kwargs']['lr_decay_rate'] ** steps)

    elif cfg['scheduler'] == 'poly':
        lambd = pow(1-(epoch/cfg['epochs']), 0.9)
        lr = lr * lambd

    else:
        raise ValueError('Invalid learning rate schedule {}'.format(cfg['scheduler']))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return lr
