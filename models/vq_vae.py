import torch
from models import BaseVAE
from torch import nn
from torch.nn import functional as F
from .types_ import *

class VectorQuantizer(nn.Module):
    """
    Reference:
    [1] https://github.com/deepmind/sonnet/blob/v2/sonnet/src/nets/vqvae.py

    将编码器输出的连续潜在向量 (latents) 映射到码本中最接近的离散向量 (量化过程)
    关键操作：
        量化机制: 计算潜在向量与所有码本向量的 L2 距离，选择距离最近的码本向量作为量化结果；
        损失设计: 用 “承诺损失” 约束编码器、“嵌入损失” 约束码本，确保两者协同学习
        梯度技巧: 通过截断梯度 (straight-through estimator) 让量化过程可训练
    """
    def __init__(self,
                 num_embeddings: int,
                 embedding_dim: int,
                 beta: float = 0.25):
        super(VectorQuantizer, self).__init__()
        self.K = num_embeddings # 码本中向量的数量（离散类别数）
        self.D = embedding_dim # 每个码本向量的维度
        self.beta = beta

        self.embedding = nn.Embedding(self.K, self.D)
        # 将数据填充为[a, b]之间均匀分布的随机数
        self.embedding.weight.data.uniform_(-1 / self.K, 1 / self.K)

    def forward(self, latents: Tensor) -> Tensor:
        """
        将编码器输出的连续潜在向量 (latents) 量化为码本中最接近的离散向量。
        Args:
            latents: 编码器输出的连续潜在特征图，形状为[B, D, H, W]
                     (B:批量大小, D:特征维度, H/W: 空间尺寸）
        Returns:
            quantized_latents: 量化后的离散特征图, 形状同latents [B, D, H, W]
            vq_loss: 量化损失，包含承诺损失和嵌入损失，用于优化编码器和码本
        """
        latents = latents.permute(0, 2, 3, 1).contiguous()  # [B x D x H x W] -> [B x H x W x D]
        latents_shape = latents.shape
        flat_latents = latents.view(-1, self.D)  # [BHW x D]

        # Compute L2 distance between latents and embedding weights
        # 计算每个潜在向量与所有码本向量的L2距离
        # 公式推导：L2距离的平方 = ||x - y||² = x² + y² - 2xy; 这里省略开方，因为比较距离大小时平方不影响结果，加速计算
        # 广播机制相加: [BHW, 1](每个潜在向量对应一个标量) + [K](每个码本向量对应一个标量) + [BHW, K]
        # 最终 dist 形状为 [BHW, K]，表示每个潜在向量与K个码本向量的L2距离平方
        dist = torch.sum(flat_latents ** 2, dim=1, keepdim=True) + \
               torch.sum(self.embedding.weight ** 2, dim=1) - \
               2 * torch.matmul(flat_latents, self.embedding.weight.t())  # [BHW x K] = [BHW x D] * [D x K]

        # Get the encoding that has the min distance
        # argmin返回每个潜在向量对应的最小距离码本索引，形状为[BHW, 1]
        encoding_inds = torch.argmin(dist, dim=1).unsqueeze(1)  # [BHW, 1]

        # Convert to one-hot encodings
        # 将索引转换为独热编码（one-hot encoding）形状为[BHW, K]，只有最小距离的索引位置为1，其余为0
        device = latents.device
        encoding_one_hot = torch.zeros(encoding_inds.size(0), self.K, device=device)
        encoding_one_hot.scatter_(1, encoding_inds, 1)  # [BHW x K]

        # Quantize the latents
        # 量化：通过矩阵乘法从码本中取出对应的离散向量
        # 独热编码与码本向量矩阵相乘 → 每个潜在向量被替换为最近的码本向量
        # 形状：[BHW, K] × [K, D] → [BHW, D]
        quantized_latents = torch.matmul(encoding_one_hot, self.embedding.weight)  # [BHW, D]
        quantized_latents = quantized_latents.view(latents_shape)  # [B x H x W x D]

        # Compute the VQ Losses
        # 承诺损失（commitment loss）：约束编码器输出接近码本向量（梯度仅流向编码器）
        # detach()切断码本向量的梯度，确保损失只优化编码器
        commitment_loss = F.mse_loss(quantized_latents.detach(), latents)
        # 嵌入损失（embedding loss）：约束码本向量接近编码器输出（梯度仅流向码本）
        # detach()切断编码器输出的梯度，确保损失只优化码本
        embedding_loss = F.mse_loss(quantized_latents, latents.detach())

        # 量化损失 = β×承诺损失+嵌入损失
        vq_loss = commitment_loss * self.beta + embedding_loss

        # Add the residue back to the latents
        # 截断梯度技巧（straight-through estimator）：解决量化过程梯度不可导问题
        # 前向保留量化真实值(最终输出的仍是真实的量化向量, 因为加了量化变化量)、反向强制梯度截断，让量化后的向量对编码器输出的梯度 “近似为 1”，从而实现梯度从解码器向编码器的传递。
        # 即：编码器梯度 = 原始潜在向量的梯度（忽略量化的离散跳变）, d(quantized_latents) / d(latents) = 1 + 0 = 1
        quantized_latents = latents + (quantized_latents - latents).detach()

        # 恢复维度顺序：[B, H, W, D] → [B, D, H, W]，匹配解码器输入格式
        return quantized_latents.permute(0, 3, 1, 2).contiguous(), vq_loss  # [B x D x H x W]

class ResidualLayer(nn.Module):
    '''
        VQ-VAE 需要提取足够抽象的特征才能实现有效量化 (否则量化后的特征语义不明确), 因此需要较深的网络。
        但深层网络易出现梯度消失, 而残差连接 (input + resblock(input)）可让梯度直接通过捷径传播，既能保证特征提取的深度（捕捉高阶语义），又能稳定训练过程。
    '''

    def __init__(self,
                 in_channels: int,
                 out_channels: int):
        super(ResidualLayer, self).__init__()
        self.resblock = nn.Sequential(nn.Conv2d(in_channels, out_channels,
                                                kernel_size=3, padding=1, bias=False),
                                      nn.ReLU(True),
                                      nn.Conv2d(out_channels, out_channels,
                                                kernel_size=1, bias=False))

    def forward(self, input: Tensor) -> Tensor:
        return input + self.resblock(input)


class VQVAE(BaseVAE):
    '''
        VAE: 潜在空间是连续的，任意两点之间的插值是有意义的，但可能存在 “语义不连续” 区域（采样结果无意义）；
        VQ-VAE: 潜在空间是离散的（由码本向量构成），只有码本中的向量或其组合是有效的，因此生成时需从码本中选择向量，但离散性更利于学习 “可解释的语义单元”。
    '''

    def __init__(self,
                 in_channels: int,
                 embedding_dim: int,
                 num_embeddings: int,
                 hidden_dims: List = None,
                 beta: float = 0.25,
                 img_size: int = 64,
                 **kwargs) -> None:
        super(VQVAE, self).__init__()

        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.img_size = img_size
        self.beta = beta

        modules = []
        if hidden_dims is None:
            hidden_dims = [128, 256]

        # Build Encoder
        # 空间下采样与通道扩张： 将输入图像从 [B, 3, 64, 64] 逐步下采样， 最终得到 [B, 256, 16, 16] 的特征图
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels=h_dim,
                              kernel_size=4, stride=2, padding=1),
                    nn.LeakyReLU())
            )
            in_channels = h_dim

        # 统一特征分布
        modules.append(
            nn.Sequential(
                nn.Conv2d(in_channels, in_channels,
                          kernel_size=3, stride=1, padding=1),
                nn.LeakyReLU())
        )
        # 残差块堆叠（深度特征提取）
        for _ in range(6):
            modules.append(ResidualLayer(in_channels, in_channels))
        modules.append(nn.LeakyReLU())

        # 将特征通道数与码本向量的维度（D）对齐，为后续向量量化（VectorQuantizer）做准备
        # 输出[B, D, 16, 16]的待量化特征图（D=embedding_dim）
        modules.append(
            nn.Sequential(
                nn.Conv2d(in_channels, embedding_dim,
                          kernel_size=1, stride=1),
                nn.LeakyReLU())
        )

        self.encoder = nn.Sequential(*modules)

        self.vq_layer = VectorQuantizer(num_embeddings,
                                        embedding_dim,
                                        self.beta)

        # Build Decoder
        modules = []
        # 将量化特征的通道数恢复到编码器残差块的输入维度（256）
        # [B, D, 16, 16] -> [B, 256, 16, 16]
        modules.append(
            nn.Sequential(
                nn.Conv2d(embedding_dim,
                          hidden_dims[-1],
                          kernel_size=3,
                          stride=1,
                          padding=1),
                nn.LeakyReLU())
        )

        # 特征细化: 对量化后的特征进行 “修复” 和 “增强”—— 量化过程会损失部分细节，残差块可通过非线性变换补充特征信息，提升重建质量
        for _ in range(6):
            modules.append(ResidualLayer(hidden_dims[-1], hidden_dims[-1]))

        modules.append(nn.LeakyReLU())

        hidden_dims.reverse()
        # 空间上采样与通道还原, [B, 256, 16, 16] -> [B, 128, 32, 32]
        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(hidden_dims[i],
                                       hidden_dims[i + 1],
                                       kernel_size=4,
                                       stride=2,
                                       padding=1),
                    nn.LeakyReLU())
            )
        # 通道数还原为RGB, 像素值约束在[-1, 1]（与输入图像范围一致）, [B, 128, 32, 32] -> [B, 3, 64, 64]
        modules.append(
            nn.Sequential(
                nn.ConvTranspose2d(hidden_dims[-1],
                                   out_channels=3,
                                   kernel_size=4,
                                   stride=2, padding=1),
                nn.Tanh()))

        self.decoder = nn.Sequential(*modules)

    def encode(self, input: Tensor) -> List[Tensor]:
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tensor) List of latent codes [N x D x H/4 x W/4]
        """
        result = self.encoder(input)
        return [result]

    def decode(self, z: Tensor) -> Tensor:
        """
        解码器能将 “码本中的向量组合” 映射为有意义的样本
        Maps the given latent codes
        onto the image space.
        :param z: (Tensor) [B x D x H x W]
        :return: (Tensor) [B x C x H x W]
        """

        result = self.decoder(z)
        return result

    def forward(self, input: Tensor, **kwargs) -> List[Tensor]:
        encoding = self.encode(input)[0]
        # 量化操作：连续向量 -> 离散码本向量
        quantized_inputs, vq_loss = self.vq_layer(encoding)
        return [self.decode(quantized_inputs), input, vq_loss]

    def loss_function(self,
                      *args,
                      **kwargs) -> dict:
        """
        :param args:
        :param kwargs:
        :return:
        """
        recons = args[0]
        input = args[1]
        vq_loss = args[2]

        recons_loss = F.mse_loss(recons, input)

        # 量化损失约束潜在向量接近码本向量
        loss = recons_loss + vq_loss
        return {'loss': loss,
                'Reconstruction_Loss': recons_loss,
                'VQ_Loss':vq_loss}

    def sample(self,
               num_samples: int,
               current_device: Union[int, str], **kwargs) -> Tensor:
        raise Warning('VQVAE sampler is not implemented.')

    def generate(self, x: Tensor, **kwargs) -> Tensor:
        """
        Given an input image x, returns the reconstructed image
        :param x: (Tensor) [B x C x H x W]
        :return: (Tensor) [B x C x H x W]
        """

        return self.forward(x)[0]