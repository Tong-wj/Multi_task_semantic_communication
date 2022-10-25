#
# Authors: Simon Vandenhende
# Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)

import argparse
import cv2
import os
import numpy as np
import sys
import torch

from utils.config import create_config
from utils.common_config import get_train_dataset, get_transformations,\
                                get_val_dataset, get_train_dataloader, get_val_dataloader,\
                                get_optimizer, get_model, adjust_learning_rate,\
                                get_criterion
from utils.logger import Logger
from train.train_utils import train_vanilla
from evaluation.evaluate_utils import eval_model, validate_results, save_model_predictions,\
                                    eval_all_results
from termcolor import colored

# Parser
parser = argparse.ArgumentParser(description='Vanilla Training')
parser.add_argument('--config_env',
                    help='Config file for the environment')
parser.add_argument('--config_exp',
                    help='Config file for the experiment')
args = parser.parse_args()

def main():
    # Retrieve config file
    cv2.setNumThreads(0)
    cfg = create_config(args.config_env, args.config_exp)
    sys.stdout = Logger(os.path.join(cfg['output_dir'], 'log_file.txt'))
    print(colored(cfg, 'red'))

    # Get model
    print(colored('Retrieve model', 'blue'))
    model = get_model(cfg)
    model = torch.nn.DataParallel(model)
    model = model.cuda()

    # Get criterion
    print(colored('Get loss', 'blue'))
    criterion = get_criterion(cfg)
    criterion.cuda()
    print(criterion)

    # CUDNN
    print(colored('Set CuDNN benchmark', 'blue')) 
    torch.backends.cudnn.benchmark = True

    # Optimizer
    print(colored('Retrieve optimizer', 'blue'))
    optimizer = get_optimizer(cfg, model)
    print(optimizer)

    # Dataset
    print(colored('Retrieve dataset', 'blue'))
    
    # Transforms 
    train_transforms, val_transforms = get_transformations(cfg)
    train_dataset = get_train_dataset(cfg, train_transforms)
    val_dataset = get_val_dataset(cfg, val_transforms)
    true_val_dataset = get_val_dataset(cfg, None) # True validation dataset without reshape 
    train_dataloader = get_train_dataloader(cfg, train_dataset)
    val_dataloader = get_val_dataloader(cfg, val_dataset)
    print('Train samples %d - Val samples %d' %(len(train_dataset), len(val_dataset)))
    print('Train transformations:')
    print(train_transforms)
    print('Val transformations:')
    print(val_transforms)
    
    # Resume from checkpoint
    if os.path.exists(cfg['checkpoint']):
        print(colored('Restart from checkpoint {}'.format(cfg['checkpoint']), 'blue'))
        checkpoint = torch.load(cfg['checkpoint'], map_location='cpu')
        optimizer.load_state_dict(checkpoint['optimizer'])
        model.load_state_dict(checkpoint['model'])
        start_epoch = checkpoint['epoch']
        best_result = checkpoint['best_result']

    else:
        print(colored('No checkpoint file at {}'.format(cfg['checkpoint']), 'blue'))
        start_epoch = 0
        save_model_predictions(cfg, val_dataloader, model)
        best_result = eval_all_results(cfg)
    
    
    # Main loop
    print(colored('Starting main loop', 'blue'))

    for epoch in range(start_epoch, cfg['epochs']):
        print(colored('Epoch %d/%d' %(epoch+1, cfg['epochs']), 'yellow'))
        print(colored('-'*10, 'yellow'))

        # Adjust lr
        lr = adjust_learning_rate(cfg, optimizer, epoch)
        print('Adjusted learning rate to {:.5f}'.format(lr))

        # Train 
        print('Train ...')
        eval_train = train_vanilla(cfg, train_dataloader, model, criterion, optimizer, epoch)

        # Evaluate
            # Check if need to perform eval first
        if 'eval_final_10_epochs_only' in cfg.keys() and cfg['eval_final_10_epochs_only']: # To speed up -> Avoid eval every epoch, and only test during final 10 epochs.
            if epoch + 1 > cfg['epochs'] - 10:
                eval_bool = True
            else:
                eval_bool = False
        else:
            eval_bool = True

        # Perform evaluation
        if eval_bool:
            print('Evaluate ...')
            save_model_predictions(cfg, val_dataloader, model)
            curr_result = eval_all_results(cfg)
            improves, best_result = validate_results(cfg, curr_result, best_result)
            if improves:
                print('Save new best model')
                torch.save(model.state_dict(), cfg['best_model'])

        # Checkpoint
        print('Checkpoint ...')
        torch.save({'optimizer': optimizer.state_dict(), 'model': model.state_dict(), 
                    'epoch': epoch + 1, 'best_result': best_result}, cfg['checkpoint'])

    # Evaluate best model at the end
    print(colored('Evaluating best model at the end', 'blue'))
    model.load_state_dict(torch.load(cfg['checkpoint'])['model'])
    save_model_predictions(cfg, val_dataloader, model)
    eval_stats = eval_all_results(cfg)

if __name__ == "__main__":
    main()
