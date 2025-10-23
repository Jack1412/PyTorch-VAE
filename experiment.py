import os
import math
import torch
from torch import optim
from models import BaseVAE
from models.types_ import *
from utils import data_loader
import pytorch_lightning as pl
from torchvision import transforms
import torchvision.utils as vutils
from torchvision.datasets import CelebA
from torch.utils.data import DataLoader


class VAEXperiment(pl.LightningModule):

    def __init__(self,
                 vae_model: BaseVAE,
                 params: dict) -> None:
        super(VAEXperiment, self).__init__()

        self.model = vae_model
        self.params = params
        self.curr_device = None
        # 是否保留计算图 (用于特定场景如对抗训练, 默认关闭) 
        self.hold_graph = False
        try:
            self.hold_graph = self.params['retain_first_backpass']
        except:
            pass

    def forward(self, input: Tensor, **kwargs) -> Tensor:
        """
        PL框架的forward方法: 定义模型的前向传播逻辑, 与PyTorch的forward一致。
        此处直接调用传入VAE模型的forward方法, 保证模型逻辑统一。
        Args:
            input: 输入数据 (如图像张量 [B, C, H, W]) 
            **kwargs: 额外参数 (如CVAE需要的labels标签) 
        Returns:
            Tensor: 模型前向输出 (如VAE的[重建图, 原始图, mu, log_var]) 
        """
        return self.model(input, **kwargs)

    def training_step(self, batch, batch_idx):
        """
        PL框架的训练步骤: 每次迭代 (处理一个batch) 的训练逻辑, 含数据读取、损失计算、日志记录。
        Args:
            batch: 训练数据批次 (含输入图像和标签, 格式为(real_img, labels)) 
            batch_idx: 当前batch的索引 (用于特殊场景如梯度累积, 此处未实际使用) 
            optimizer_idx: 优化器索引 (用于多优化器场景如对抗训练, 默认用第0个优化器) 
        Returns:
            Tensor: 训练总损失 (PL会自动根据该损失反向传播更新参数) 
        """
        real_img, labels = batch
        self.curr_device = real_img.device

        results = self.forward(real_img, labels = labels)
        # M_N: KL损失的权重 (通常为"当前batch大小/训练集总样本数", 控制KL损失的量级) 
        train_loss = self.model.loss_function(*results,
                                              M_N = self.params['kld_weight'], #al_img.shape[0]/ self.num_train_imgs,
                                            #   optimizer_idx=optimizer_idx,
                                              batch_idx = batch_idx)

        # self.log_dict 是 LightningModule 提供的日志记录接口，用于将多个键值对形式的指标（如损失、准确率）同时记录到日志器（如 TensorBoard）
        # item() 将 {'loss': tensor(0.5), 'Reconstruction_Loss': tensor(0.3), 'KLD': tensor(0.2)} 转换为
        # {'loss': 0.5, 'Reconstruction_Loss': 0.3, 'KLD': 0.2}
        self.log_dict({key: val.item() for key, val in train_loss.items()}, sync_dist=True)

        return train_loss['loss']

    def validation_step(self, batch, batch_idx):
        """
        PL框架的验证步骤: 每次验证迭代 (处理一个batch) 的逻辑, 与训练步骤类似但不更新参数。
        作用: 评估模型在验证集上的性能, 避免过拟合。
        Args:
            同training_step (batch为验证集数据) 
        """
        real_img, labels = batch
        self.curr_device = real_img.device

        # 模型前向传播 (验证阶段不启用Dropout等训练特有的层, PL会自动处理model.eval()) 
        results = self.forward(real_img, labels = labels)
        # 计算验证损失: M_N设为1.0 (验证阶段通常不调整KL权重, 直接用原始损失) 
        val_loss = self.model.loss_function(*results,
                                            M_N = 1.0, #real_img.shape[0]/ self.num_val_imgs,
                                            # optimizer_idx = optimizer_idx,
                                            batch_idx = batch_idx)
        # 日志记录: 验证损失的键名加"val_"前缀 (如val_loss、val_Reconstruction_Loss) , 避免与训练日志混淆
        self.log_dict({f"val_{key}": val.item() for key, val in val_loss.items()}, sync_dist=True)

        
    def on_validation_end(self) -> None:
        """
        PL框架的验证结束钩子: 在整个验证阶段 (所有验证batch处理完) 后执行。
        作用: 生成并保存"重建图像"和"生成图像", 可视化模型性能。
        """
        self.sample_images()
        
    def sample_images(self):
        # Get sample reconstruction image
        # 获取验证集数据 (从PL的DataModule中获取测试/验证数据加载器, 取第一个batch)  
        # fit()时, PL 内部会将 Trainer 自身（runner）赋值给 experiment 的 trainer 属性，即 experiment.trainer = runner。       
        test_input, test_label = next(iter(self.trainer.datamodule.test_dataloader()))
        test_input = test_input.to(self.curr_device)
        test_label = test_label.to(self.curr_device)

        # test_input, test_label = batch
        # 生成重建图像: 调用模型的generate方法 (输入原始图像, 输出重建图像) 
        recons = self.model.generate(test_input, labels = test_label)
        # 保存重建图像: 用vutils.save_image拼接图像, 按" epoch号"命名, 保存在日志目录的Reconstructions文件夹
        vutils.save_image(recons.data,
                          os.path.join(self.logger.log_dir , 
                                       "Reconstructions", 
                                       f"recons_{self.logger.name}_Epoch_{self.current_epoch}.png"),
                          normalize=True, # 图像归一化 (将像素值从[-1,1]映射到[0,1]) 
                          nrow=12)

        # 生成新样本: 调用模型的sample方法
        try:
            samples = self.model.sample(144,
                                        self.curr_device,
                                        labels = test_label)
            vutils.save_image(samples.cpu().data,
                              os.path.join(self.logger.log_dir , 
                                           "Samples",      
                                           f"{self.logger.name}_Epoch_{self.current_epoch}.png"),
                              normalize=True,
                              nrow=12)
        except Warning:
            # VQ-VAE模型没有sample方法
            pass

    def configure_optimizers(self):
        """
        PL框架的优化器配置方法: 定义训练使用的优化器和学习率调度器。
        支持单优化器、多优化器 (如对抗训练的生成器/判别器优化器) 和学习率衰减。
        Returns:
            优化器列表 + 调度器列表 (PL会自动管理优化器的step和调度器的step) 
        """

        optims = []
        scheds = []

        # 配置主优化器
        optimizer = optim.Adam(self.model.parameters(),
                               lr=self.params['LR'],
                               weight_decay=self.params['weight_decay'])
        optims.append(optimizer)
        # Check if more than 1 optimizer is required (Used for adversarial training)
        try:
            if self.params['LR_2'] is not None:
                optimizer2 = optim.Adam(getattr(self.model,self.params['submodel']).parameters(),
                                        lr=self.params['LR_2'])
                optims.append(optimizer2)
        except:
            pass

        # 配置学习率调度器 (如指数衰减, 防止后期训练震荡) 
        try:
            if self.params['scheduler_gamma'] is not None:
                # 为主优化器配置指数衰减调度器 (lr = lr * gamma^epoch) 
                scheduler = optim.lr_scheduler.ExponentialLR(optims[0],
                                                             gamma = self.params['scheduler_gamma'])
                scheds.append(scheduler)

                # Check if another scheduler is required for the second optimizer
                try:
                    if self.params['scheduler_gamma_2'] is not None:
                        scheduler2 = optim.lr_scheduler.ExponentialLR(optims[1],
                                                                      gamma = self.params['scheduler_gamma_2'])
                        scheds.append(scheduler2)
                except:
                    pass
                # 返回优化器列表和调度器列表 (PL会自动在每个epoch后更新调度器) 
                print("len of optims: ", len(optims))
                print("len of scheds: ", len(scheds))
                return optims, scheds
        except:
            return optims
