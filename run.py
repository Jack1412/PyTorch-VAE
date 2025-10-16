import os
import yaml
import argparse
import numpy as np
from pathlib import Path
from models import *
from experiment import VAEXperiment
import torch.backends.cudnn as cudnn
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.utilities.seed import seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from dataset import VAEDataset
from pytorch_lightning.plugins import DDPPlugin


parser = argparse.ArgumentParser(description='Generic runner for VAE models')
parser.add_argument('--config',  '-c',
                    dest="filename",
                    metavar='FILE',
                    help =  'path to the config file',
                    default='configs/vae.yaml')

args = parser.parse_args()
with open(args.filename, 'r') as file:
    try:
        config = yaml.safe_load(file)
    except yaml.YAMLError as exc:
        print(exc)


tb_logger =  TensorBoardLogger(save_dir=config['logging_params']['save_dir'],
                               name=config['model_params']['name'],)

# For reproducibility
seed_everything(config['exp_params']['manual_seed'], True)

# vae_models是models/__init__.py中定义的字典，格式如 {VanillaVAE: VanillaVAE, CVAE: CVAE}
model = vae_models[config['model_params']['name']](**config['model_params'])
experiment = VAEXperiment(model,
                          config['exp_params'])

data = VAEDataset(**config["data_params"], pin_memory=len(config['trainer_params']['gpus']) != 0)

# PL DataModule 的核心方法，在每个进程中执行一次，完成数据集划分（显式调用 setup() 不是必需的，但无害，runner.fit会自动调用 setup()）
data.setup()
runner = Trainer(logger=tb_logger,
                 callbacks=[
                     LearningRateMonitor(),
                     ModelCheckpoint(save_top_k=2, # 保存验证集损失Top-2的模型（最优和次优）
                                     dirpath =os.path.join(tb_logger.log_dir , "checkpoints"), 
                                     monitor= "val_loss", # 监控的指标（以验证集损失为标准）
                                     save_last= True), # 额外保存最后一个epoch的模型（方便断点续训）
                 ],
                 strategy=DDPPlugin(find_unused_parameters=False),
                 **config['trainer_params'])

# 创建样本保存目录, 重建图保存目录
Path(f"{tb_logger.log_dir}/Samples").mkdir(exist_ok=True, parents=True)
Path(f"{tb_logger.log_dir}/Reconstructions").mkdir(exist_ok=True, parents=True)


print(f"======= Training {config['model_params']['name']} =======")
# 1. 加载 data 的训练 / 验证数据加载器（train_dataloader()、val_dataloader()）；
# 2. 按 max_epochs 循环训练，每轮执行 experiment.training_step() 和 experiment.validation_step()；
# 3. 自动处理梯度清零、反向传播、参数更新、日志记录、模型保存等；
runner.fit(experiment, datamodule=data)