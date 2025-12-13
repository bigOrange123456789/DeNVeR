import torch
from torch import nn
import numpy as np

# class AdaptivePositionalEncoder(nn.Module):
#     """
#     自适应位置编码器，在训练过程中动态调整编码频率
#     基于《Few-shot NeRF by Adaptive Rendering Loss Regularization》的思想
#     """
    
#     def __init__(self, input_dim=3, num_freqs=10, include_input=True, 
#                  total_steps=2000, warmup_steps=1500, mode='linear'):
#         """
#         Args:
#             input_dim: 输入维度(如3D坐标的维度为3)
#             num_freqs: 位置编码的频率数量
#             include_input: 是否包含原始输入
#             total_steps: 总训练步数
#             warmup_steps: 热身步数，在此期间频率逐渐增加
#             mode: 频率调整模式 ('linear', 'cosine', 'exponential') #线性遮挡,小于0的为0
#         """
#         super().__init__()
#         self.input_dim = input_dim
#         self.num_freqs = num_freqs
#         self.include_input = include_input
#         self.total_steps = total_steps #总步数输入进去有什么用?
#         self.warmup_steps = warmup_steps
#         self.mode = mode
        
#         # 预计算频率带宽（几何级数）
#         self.freq_bands = 2.0 ** torch.linspace(0, num_freqs-1, num_freqs)
        
#         # 注册为buffer，这些张量会被保存到模型状态中但不参与梯度更新
#         self.register_buffer('current_step', torch.tensor(0))
        
#         # 计算输出维度
#         self.output_dim = input_dim * (2 * num_freqs + include_input)
        
#     def get_frequency_mask(self, current_step=None):
#         """
#         根据当前训练步数计算频率掩码
#         Returns:
#             freq_mask: 形状为 [num_freqs] 的掩码张量，1表示启用，0表示遮挡
#         """
#         if current_step is None:
#             current_step = self.current_step
            
#         # 确保步数在合理范围内
#         current_step = min(current_step, self.total_steps)
        
#         if current_step >= self.warmup_steps:
#             # 热身阶段结束，启用所有频率
#             return torch.ones(self.num_freqs, device=self.freq_bands.device)
        
#         # 计算当前启用的频率比例
#         progress = current_step / self.warmup_steps
        
#         if self.mode == 'linear':
#             enabled_ratio = progress
#         elif self.mode == 'cosine':
#             enabled_ratio = 0.5 * (1 - torch.cos(torch.tensor(progress * np.pi)))
#         elif self.mode == 'exponential':
#             enabled_ratio = 1 - torch.exp(-5 * progress)
#         else:
#             enabled_ratio = progress
        
#         # 计算应该启用的频率数量
#         num_enabled_freqs = int(enabled_ratio * self.num_freqs)
        
#         # 创建掩码（启用低频，遮挡高频）
#         freq_mask = torch.zeros(self.num_freqs, device=self.freq_bands.device)
#         freq_mask[:num_enabled_freqs] = 1.0
        
#         return freq_mask
    
#     def forward(self, x, current_step=None):
#         """
#         Args:
#             x: 输入张量，形状为 [..., input_dim]
#             current_step: 当前训练步数（可选）
#         Returns:
#             encoded: 编码后的张量，形状为 [..., output_dim]
#         """
#         if current_step is not None:
#             self.current_step = torch.tensor(current_step, device=x.device)
        
#         # 获取当前步数的频率掩码
#         freq_mask = self.get_frequency_mask()
        
#         # 如果设置为包含原始输入，则将其加入编码列表
#         encoded = [x] if self.include_input else []
        
#         # 对每个频率应用位置编码，并根据掩码进行遮挡
#         for i, freq in enumerate(self.freq_bands):
#             # 应用频率掩码
#             mask_val = freq_mask[i] #值域范围[0,1]
#             if mask_val > 0:
#                 # 启用该频率
#                 encoded.append(torch.sin(freq * x) * mask_val)
#                 encoded.append(torch.cos(freq * x) * mask_val)
#             else:
#                 # 遮挡该频率：添加零张量以保持输出维度一致
#                 zero_sin = torch.zeros_like(x)
#                 zero_cos = torch.zeros_like(x)
#                 encoded.append(zero_sin)
#                 encoded.append(zero_cos)
        
#         # 沿最后一个维度拼接所有编码后的特征
#         return torch.cat(encoded, dim=-1)
    
#     def get_current_freq_info(self): #似乎是用于测试
#         """返回当前启用的频率信息，用于监控训练过程"""
#         freq_mask = self.get_frequency_mask()
#         enabled_freqs = torch.sum(freq_mask).item()
#         return {
#             'current_step': self.current_step.item(),
#             'enabled_freqs': enabled_freqs,
#             'total_freqs': self.num_freqs,
#             'enabled_ratio': enabled_freqs / self.num_freqs,
#             'freq_mask': freq_mask.detach().cpu().numpy()
#         }
    
# class PositionalEncoder(nn.Module): #空间编码+视频编码
#     """位置编码，将低维输入映射到高维空间以捕获高频细节。"""
#     def __init__(self, input_dim, num_freqs, include_input=True):
#         super().__init__()
#         self.include_input = include_input
#         self.num_freqs = num_freqs
#         self.output_dim = input_dim * (2 * num_freqs + include_input)
#         # 生成频率带宽，使用2的幂次方
#         self.freq_bands = 2.0 ** torch.linspace(0, num_freqs-1, num_freqs)

#     def forward(self, x):
#         """
#         对输入应用位置编码。
#         Args:
#             x: 输入张量，形状为 [..., input_dim]
#         Returns:
#             encoded: 编码后的张量，形状为 [..., output_dim]
#         """
#         # 初始化编码列表，如果设置为包含原始输入，则将其加入
#         encoded = [x] if self.include_input else []
#         for freq in self.freq_bands:
#             # 对每个频率分别计算正弦和余弦值
#             encoded.append(torch.sin(freq * x))
#             encoded.append(torch.cos(freq * x))
#         # 沿最后一个维度拼接所有编码后的特征
#         return torch.cat(encoded, dim=-1)

class SineLayer(nn.Module):
    # See paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of omega_0.
    
    # If is_first=True, omega_0 is a frequency factor which simply multiplies the activations before the 
    # nonlinearity. Different signals may require different omega_0 in the first layer - this is a 
    # hyperparameter.
    
    # If is_first=False, then the weights will be divided by omega_0 so as to keep the magnitude of 
    # activations constant, but boost gradients to the weight matrix (see supplement Sec. 1.5)
    
    def __init__(self, in_features, out_features, bias=True,
                 is_first=False, omega_0=30):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first
        
        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        
        self.init_weights()
    
    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features, 
                                             1 / self.in_features)
            else:
                self.linear.weight.uniform_(-np.sqrt(6 / self.in_features) / self.omega_0, 
                                             np.sqrt(6 / self.in_features) / self.omega_0)
        
    def forward(self, input):
        return torch.sin(self.omega_0 * self.linear(input))
    
    def forward_with_intermediate(self, input): #这个函数并没有被执行
        if True:
            print("我认为这个函数并没有被执行:nir/model.py/class SineLayer/forward_with_intermediate()")
            exit(0)
        # For visualization of activation distributions
        intermediate = self.omega_0 * self.linear(input)
        return torch.sin(intermediate), intermediate

class Siren(nn.Module):
    # def __init__(self, in_features, hidden_features, hidden_layers, out_features, outermost_linear=False, 
    #              first_omega_0=30, hidden_omega_0=30.):
    def __init__(self, in_features, hidden_features, hidden_layers, out_features, outermost_linear=False, 
                 first_omega_0=30, hidden_omega_0=30., 
                 use_residual=False, 
                #  use_positionEncoder=False, total_steps=2000, warmup_steps=1500, 
                 ):
        super().__init__()
        
        # self.use_positionEncoder = use_positionEncoder
        # if self.use_positionEncoder:
        #     # 自适应位置编码器
        #     self.pos_encoder = AdaptivePositionalEncoder(
        #         input_dim=3, num_freqs=10, 
        #         total_steps=total_steps, warmup_steps=warmup_steps, 
        #         mode='linear'
        #     )
            
            # # 方向编码器（也可以改为自适应，但通常方向编码不需要）
            # self.dir_encoder = AdaptivePositionalEncoder(
            #     input_dim=3, num_freqs=4, total_steps=total_steps,
            #     warmup_steps=warmup_steps, mode='linear'
            # )

        self.net = []
        # 1.输入层
        self.net.append(SineLayer(in_features, hidden_features, 
                                  is_first=True, omega_0=first_omega_0))

        # 2.隐含层
        for i in range(hidden_layers):
            self.net.append(SineLayer(hidden_features, hidden_features, 
                                      is_first=False, omega_0=hidden_omega_0))

        # 3.输出层
        if outermost_linear:
            final_linear = nn.Linear(hidden_features, out_features)
            
            with torch.no_grad():
                final_linear.weight.uniform_(-np.sqrt(6 / hidden_features) / hidden_omega_0, 
                                              np.sqrt(6 / hidden_features) / hidden_omega_0)
                
            self.net.append(final_linear)
        else:
            self.net.append(SineLayer(hidden_features, out_features, 
                                      is_first=False, omega_0=hidden_omega_0))
        
        self.net = nn.Sequential(*self.net)
        self.use_residual=use_residual
    
    def forward_old(self, coords):
        output = self.net(coords)
        return output

    def forward(self, coords):
        if not self.use_residual:
            return self.forward_old(coords)
        
        x = coords
        
        # 通过输入层
        x = self.net[0](x)
        
        # 通过隐含层（从索引1开始）
        for i in range(1, len(self.net) - 1):  # 排除输入层和输出层
            # 残差连接的起点
            residual = x
            
            # 通过当前层（预激活：先激活再线性变换）
            # 注意：Siren的SineLayer已经是sin(omega_0 * linear(x))形式
            # 所以这里我们直接使用当前层
            x = self.net[i](x)
            
            # 添加残差连接
            x = x + residual
        
        # 通过输出层
        if len(self.net) > 1:
            x = self.net[-1](x)
        
        return x


class Homography(nn.Module):
    def __init__(self, in_features=1, hidden_features=256, hidden_layers=1):
        super().__init__()
        out_features = 8
        
        self.net = []
        self.net.append(nn.Linear(in_features, hidden_features))
        self.net.append(nn.ReLU(inplace=True))
        for i in range(hidden_layers):
            self.net.append(nn.Linear(hidden_features, hidden_features))
            self.net.append(nn.ReLU(inplace=True))
        self.net.append(nn.Linear(hidden_features, out_features))     
        self.net = nn.Sequential(*self.net)
        
        self.init_weights()
        
    def init_weights(self):
        with torch.no_grad():
            self.net[-1].bias.copy_(torch.Tensor([1., 0., 0., 0., 1., 0., 0., 0.]))
    
    def forward(self, coords):
        output = self.net(coords)
        return output