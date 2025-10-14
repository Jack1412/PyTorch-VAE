import torch
from models import BaseVAE
from torch import nn
from torch.nn import functional as F
from .types_ import *


class VanillaVAE(BaseVAE):


    def __init__(self,
                 in_channels: int,
                 latent_dim: int,
                 hidden_dims: List = None,
                 **kwargs) -> None:
        super(VanillaVAE, self).__init__()

        self.latent_dim = latent_dim

        modules = []
        if hidden_dims is None:
            hidden_dims = [32, 64, 128, 256, 512]

        # Build Encoder
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    # H_{out} = (H_{in} + 2 * padding - kernel_size) / stride + 1
                    # H_{out} 是向下取整
                    # 在本例中, (2 * padding - kernel_size) / stride = -0.5, 而像素大概率是偶数, 所以存在小数部分, 在 ConvTranspose2d 时, 需要设置 output_padding = 1
                    # 输入维度限制: 3D (unbatched) or 4D (batched)
                    # Conv2d 的 channel 数怎么变化: 
                    # 1. 每个输出通道对应一个卷积核组，该组的卷积核数量与输入通道数一致(比如in_channels=3, out_channels=64, 则有64个卷积核组, 每个组有3个卷积核)
                    # 2. 当计算一个输出通道时，卷积核组分别与输入各通道卷积，然后求和得到
                    nn.Conv2d(in_channels, out_channels=h_dim,
                              kernel_size= 3, stride= 2, padding  = 1),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU())
            )
            in_channels = h_dim

        self.encoder = nn.Sequential(*modules)
        # 通过全连接层得到 μ 和 logσ²
        self.fc_mu = nn.Linear(hidden_dims[-1]*4, latent_dim)
        self.fc_var = nn.Linear(hidden_dims[-1]*4, latent_dim)


        # Build Decoder
        modules = []
        self.decoder_input = nn.Linear(latent_dim, hidden_dims[-1] * 4)
        hidden_dims.reverse()

        # 注意, 这里少了一个反卷积层, len(hidden_dims) - 1 = 4
        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    # H_{out} = (H_{in} - 1) * stride - 2 * padding + kernel_size + output_padding
                    # channel 维度怎么处理，和卷积层的原理一样：
                    # 1. 每个输出通道对应一个卷积核组，该组的卷积核数量与输入通道数一致(比如in_channels=3, out_channels=64, 则有64个卷积核组, 每个组有3个卷积核)
                    # 2. 当计算一个输出通道时，卷积核组分别与输入各通道进行转置卷积运算，然后逐像素相加得到
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

        # 最后一个反卷积 + 变换到图片尺寸
        self.final_layer = nn.Sequential(
                            # 最终上采样到目标空间尺寸, 通道不变
                            nn.ConvTranspose2d(hidden_dims[-1],
                                               hidden_dims[-1],
                                               kernel_size=3,
                                               stride=2,
                                               padding=1,
                                               output_padding=1),
                            nn.BatchNorm2d(hidden_dims[-1]),
                            nn.LeakyReLU(),
                            # 通道变为3, 尺寸不变
                            nn.Conv2d(hidden_dims[-1], out_channels= 3,
                                      kernel_size= 3, padding= 1),
                            # 将像素值约束到 [-1, 1] 范围
                            nn.Tanh())

    def encode(self, input: Tensor) -> List[Tensor]:
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tensor) List of latent codes [N x latent_dim]
        """
        result = self.encoder(input) # [N, hidden_dims[-1],  H/32, W/32]
        result = torch.flatten(result, start_dim=1) # [N, hidden_dims[-1] * 4], 这里说明 H 和 W 必须是64才行 ???

        # Split the result into mu and var components
        # of the latent Gaussian distribution
        # log_var 是方差（σ²）的自然对数（log (σ²)），解释：
        # 为什么不直接输出方差 var，而要输出 logvar？ 因为方差的数学性质是 必须非负（σ² ≥ 0），但神经网络的输出是无界的（可能为负）。
        # 如果直接预测 var，无法保证其非负性；而预测 logvar（无界，可正可负），再通过指数运算转换为 var，就能强制保证方差非负（指数函数的结果恒正）。
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)

        return [mu, log_var]

    def decode(self, z: Tensor) -> Tensor:
        """
        Maps the given latent codes
        onto the image space.
        :param z: (Tensor) [B x D]
        :return: (Tensor) [B x C x H x W]
        """
        result = self.decoder_input(z)
        result = result.view(-1, 512, 2, 2)
        result = self.decoder(result)
        result = self.final_layer(result)
        return result

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        """
        Reparameterization trick to sample from N(mu, var) from N(0,1).
        Args:
            :param mu: (Tensor) Mean of the latent Gaussian [B x D]
            :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
        Return: 
            (Tensor) [B x D]
        """
        # sqrt(exp(logvar)) 和 exp(0.5 * logvar) 是完全等价的，但在数值计算中，exp(0.5 * logvar) 更优：
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std) # 标准正态分布

        # 连续采样
        return eps * std + mu

    def forward(self, input: Tensor, **kwargs) -> List[Tensor]:
        mu, log_var = self.encode(input)
        z = self.reparameterize(mu, log_var)
        return  [self.decode(z), input, mu, log_var]

    def loss_function(self,
                      *args,
                      **kwargs) -> dict:
        """
        Computes the VAE loss function.
        KL(N(\mu, \sigma), N(0, 1)) = \log \frac{1}{\sigma} + \frac{\sigma^2 + \mu^2}{2} - \frac{1}{2}
        KL(N(mu, sigma), N(0, 1)) = -0.5 * (1 + log_var - log_var.exp() - mu ** 2)
        :param args:
        :param kwargs:
        :return:
        """
        recons = args[0]
        input = args[1]
        mu = args[2]
        log_var = args[3]

        kld_weight = kwargs['M_N'] # Account for the minibatch samples from the dataset
        recons_loss = F.mse_loss(recons, input)

        # 对单个样本的所有潜在维度的KL散度进行求和（dim=1）
        # 对批量样本求均值（dim=0）
        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)
        
        # Loss = Reconstruction_Loss + β * KL_Loss (β <=> kld_weight)
        # Loss 平衡 “重建质量” 与 “潜在空间规整性” 这两个目标，但二者存在内在冲突
        # 最小化重建损失：希望解码器从潜在编码 z 重建出的样本与原始输入尽可能一致, 优先保留输入的细节信息，甚至可能让 z 的分布变得 “混乱”（比如不同类别的样本在潜在空间中重叠），只要能保证重建精度。
        # 最小化KL散度：希望编码器输出的 z 潜在分布 N(mu, sigma^2) 尽可能接近先验分布 N(0,1)，从而让潜在空间具备规整性（如连续性、平滑性）, 便于采样和生成; 但过度约束 KL 损失，可能会让编码器为了让 z 符合正态分布，丢弃输入的细节信息，导致重建质量下降。
        # β平衡重构损失和KL损失的量级, 因为重建损失通常是逐像素 / 逐特征计算，数值量级远大于 KL 损失, 所以 β 通常会大很多
        # 在原版的VAE中, 它通常是 1.0, 但在 beta-VAE 变体中
        '''
        β = 0: 这是一个极端情况。模型完全忽略了KL散度项, 只优化重构损失。
        Loss = Reconstruction_Loss + 0 * KL_Loss = Reconstruction_Loss
        这本质上退化成了一个自编码器 (Autoencoder) , 而不是变分自编码器。它的潜在空间将不再具有规则的结构 (不再是标准正态分布) , 因此生成能力会很差 (从 $N(0, I)$ 采样的点解码出的图像没有意义) 。

        0 < β < 1: 模型仍然优先保证重构质量, 但对潜在分布施加了较弱的约束, 使其不会完全失去规整性。有时用于解决“KL消失 (KL Vanishing) ”问题, 即模型为了完美重构而完全忽略KL项。

        β > 1: 意味着模型更侧重于对潜在分布的正则化约束。这是 β-VAE 论文的核心思想。增大β会鼓励潜在编码之间更加统计独立, 从而学习到解耦的 (disentangled)  表示 (比如某个轴表示颜色, 某个轴表示纹理)
        '''
        loss = recons_loss + kld_weight * kld_loss
        return {'loss': loss, 'Reconstruction_Loss':recons_loss.detach(), 'KLD':-kld_loss.detach()}

    def sample(self,
               num_samples:int,
               current_device: int, **kwargs) -> Tensor:
        """
        Samples from the latent space and return the corresponding
        image space map.
        :param num_samples: (Int) Number of samples
        :param current_device: (Int) Device to run the model
        :return: (Tensor)
        """
        z = torch.randn(num_samples,
                        self.latent_dim)

        z = z.to(current_device)

        samples = self.decode(z)
        return samples

    def generate(self, x: Tensor, **kwargs) -> Tensor:
        """
        Given an input image x, returns the reconstructed image
        :param x: (Tensor) [B x C x H x W]
        :return: (Tensor) [B x C x H x W]
        """

        return self.forward(x)[0]