import os
import torch
import logging
import argparse
from torch.utils.data import DataLoader
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from nanodet.util import cfg, load_config, Logger
from nanodet.trainer import TrainingTask
from nanodet.evaluator import build_evaluator
from nanodet.data.dataset import build_dataset
from nanodet.data.collate import naive_collate

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config/dota/nano-det/nanodet_dota.yml',
                        help='path to config file')
    parser.add_argument('--local_rank', type=int, default=0,
                        help='DDP parameter')
    args = parser.parse_args()
    return args

def main(args):
    # Load config
    load_config(cfg, args.config)
    
    # Create work directory
    os.makedirs(cfg.save_dir, exist_ok=True)
    
    # Setup logger
    Logger(args.local_rank, save_dir=cfg.save_dir)
    logger = logging.getLogger('nanodet')
    logger.info("Config loaded successfully!")
    
    # Create dataset
    train_dataset = build_dataset(cfg, mode="train")
    val_dataset = build_dataset(cfg, mode="val")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.device.batchsize_per_gpu,
        shuffle=True,
        num_workers=cfg.device.workers_per_gpu,
        pin_memory=True,
        collate_fn=naive_collate,
        drop_last=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.device.batchsize_per_gpu,
        shuffle=False,
        num_workers=cfg.device.workers_per_gpu,
        pin_memory=True,
        collate_fn=naive_collate,
        drop_last=False
    )
    
    # Create evaluator
    evaluator = build_evaluator(cfg.evaluator, val_dataset)
    
    # Create training task
    task = TrainingTask(cfg, evaluator)
    
    # Create trainer
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(cfg.save_dir, 'checkpoints'),
        filename='nanodet-{epoch:02d}-{val_loss:.2f}',
        save_top_k=3,
        monitor='val_loss',
        mode='min'
    )
    
    trainer = Trainer(
        max_epochs=cfg.schedule.total_epochs,
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=1,
        callbacks=[checkpoint_callback],
        default_root_dir=cfg.save_dir,
        check_val_every_n_epoch=cfg.schedule.val_intervals,
        num_sanity_val_steps=0
    )
    
    # Start training
    trainer.fit(task, train_loader, val_loader)

if __name__ == '__main__':
    args = parse_args()
    main(args) 