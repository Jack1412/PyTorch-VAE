import torch
from models import BaseVAE
from torch import nn
from torch.nn import functional as F
from .types_ import *


class ConditionalVAE(BaseVAE):
    """
        条件变分自编码器(Conditional VAE, CVAE): 在VAE基础上引入标签信息,
        #       维度                        VAE(普通变分自编码器)                           CVAE(条件变分自编码器) 
        # ----------------  ---------------------------------------------  ---------------------------------------------------------
        # 输入信息              仅输入数据(如图像 x)                                输入数据(x) + 标签(y, 如类别信息) 
        # 编码器设计            仅对数据 x 编码, 输出潜在分布 N(z;μ(x),σ²(x))         对“x + 标签嵌入”编码, 输出条件分布 N(z;μ(x,y),σ²(x,y))
        # 解码器设计            仅用潜在向量 z 解码, 重建 x                          用“z + 标签 y”解码, 重建 x(解码过程受标签控制) 
        # 生成能力              无目标生成(从 N(0,1) 随机采样 z, 生成任意样本)         可控生成(指定标签 y, 生成该标签对应的样本) 
    """

    def __init__(self,
                 in_channels: int,
                 num_classes: int,
                 latent_dim: int,
                 hidden_dims: List = None,
                 img_size:int = 64,
                 **kwargs) -> None:
        super(ConditionalVAE, self).__init__()

        self.latent_dim = latent_dim
        self.img_size = img_size

        # 标签嵌入层(将类别标签转换为与图像尺寸匹配的特征图) 
        self.embed_class = nn.Linear(num_classes, img_size * img_size)
        # 输入图像嵌入层(对输入图像做轻微特征变换, 方便与标签特征拼接) 
        self.embed_data = nn.Conv2d(in_channels, in_channels, kernel_size=1)

        modules = []
        if hidden_dims is None:
            hidden_dims = [32, 64, 128, 256, 512]

        # 编码器输入通道数+1(因为要拼接标签特征图) 
        in_channels += 1 # To account for the extra label channel
        # Build Encoder
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels=h_dim,
                              kernel_size= 3, stride= 2, padding  = 1),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU())
            )
            in_channels = h_dim

        self.encoder = nn.Sequential(*modules)
        self.fc_mu = nn.Linear(hidden_dims[-1]*4, latent_dim)
        self.fc_var = nn.Linear(hidden_dims[-1]*4, latent_dim)


        # Build Decoder
        modules = []

        # 解码器输入层：将潜在向量+标签拼接后映射到解码器初始特征
        self.decoder_input = nn.Linear(latent_dim + num_classes, hidden_dims[-1] * 4)

        hidden_dims.reverse()

        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(hidden_dims[i],
                                       hidden_dims[i + 1],
                                       kernel_size=3,
                                       stride = 2,
                                       padding=1,
                                       output_padding=1),
                    nn.BatchNorm2d(hidden_dims[i + 1]),
                    nn.LeakyReLU())
            )



        self.decoder = nn.Sequential(*modules)

        self.final_layer = nn.Sequential(
                            nn.ConvTranspose2d(hidden_dims[-1],
                                               hidden_dims[-1],
                                               kernel_size=3,
                                               stride=2,
                                               padding=1,
                                               output_padding=1),
                            nn.BatchNorm2d(hidden_dims[-1]),
                            nn.LeakyReLU(),
                            nn.Conv2d(hidden_dims[-1], out_channels= 3,
                                      kernel_size= 3, padding= 1),
                            nn.Tanh())

    def encode(self, input: Tensor) -> List[Tensor]:
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tensor) List of latent codes
        """
        result = self.encoder(input)
        result = torch.flatten(result, start_dim=1)

        # Split the result into mu and var components
        # of the latent Gaussian distribution
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)

        return [mu, log_var]

    def decode(self, z: Tensor) -> Tensor:
        """解码器：将潜在向量+标签的拼接向量映射回图像空间"""
        result = self.decoder_input(z)
        result = result.view(-1, 512, 2, 2)
        result = self.decoder(result)
        result = self.final_layer(result)
        return result

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        """
        Will a single z be enough ti compute the expectation
        for the loss??
        :param mu: (Tensor) Mean of the latent Gaussian
        :param logvar: (Tensor) Standard deviation of the latent Gaussian
        :return:
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, input: Tensor, **kwargs) -> List[Tensor]:
        '''
            CVAE 通过embed_class将标签 (如 [0,1,0] 表示类别 1) 转换为与图像同尺寸的特征图([B,1,H,W]) , 
            再与图像特征([B,C,H,W]) 在通道维度拼接, 使编码器同时感知数据和标签信息
        '''
        # 获取标签(如MNIST的数字标签) 
        y = kwargs['labels'].float()
        # 处理标签：将类别标签嵌入为与图像同尺寸的特征图
        embedded_class = self.embed_class(y)
        embedded_class = embedded_class.view(-1, self.img_size, self.img_size).unsqueeze(1)
        # 处理输入图像：轻微特征变换, 方便与标签特征拼接
        embedded_input = self.embed_data(input)
        # 拼接图像特征和标签特征(通道维度拼接) 
        x = torch.cat([embedded_input, embedded_class], dim = 1)
        mu, log_var = self.encode(x)

        z = self.reparameterize(mu, log_var)
        # 拼接z和标签(用于解码器输入) 
        z = torch.cat([z, y], dim = 1)
        return  [self.decode(z), input, mu, log_var]

    def loss_function(self,
                      *args,
                      **kwargs) -> dict:
        recons = args[0]
        input = args[1]
        mu = args[2]
        log_var = args[3]

        kld_weight = kwargs['M_N']  # Account for the minibatch samples from the dataset
        recons_loss =F.mse_loss(recons, input)

        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)

        loss = recons_loss + kld_weight * kld_loss
        return {'loss': loss, 'Reconstruction_Loss':recons_loss, 'KLD':-kld_loss}

    def sample(self,
               num_samples:int,
               current_device: int,
               **kwargs) -> Tensor:
        """
        Samples from the latent space and return the corresponding
        image space map.
        :param num_samples: (Int) Number of samples
        :param current_device: (Int) Device to run the model
        :return: (Tensor)
        """
        y = kwargs['labels'].float()
        z = torch.randn(num_samples,
                        self.latent_dim)

        z = z.to(current_device)

        z = torch.cat([z, y], dim=1)
        samples = self.decode(z)
        return samples

    def generate(self, x: Tensor, **kwargs) -> Tensor:
        """
        Given an input image x, returns the reconstructed image
        :param x: (Tensor) [B x C x H x W]
        :return: (Tensor) [B x C x H x W]
        """

        return self.forward(x, **kwargs)[0]