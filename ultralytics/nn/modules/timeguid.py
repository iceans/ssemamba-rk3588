from .tgsmamba import *
import math
import torch
import torch.nn as nn
from ultralytics.nn.modules.conv import Conv
class Split(nn.Module):
    def __init__(self,p1,p2,id):
        super(Split, self).__init__()
        self.id = id
        self.p1 = p1
        self.p2 = p2
    def forward(self, x):
        # leftnum = x.shape[1]-3
        # return torch.split(x, [self.p1,self.p2], dim=1)[self.id]
        # x = torch.split(x, [self.p1, self.p2], dim=1)[self.id]
        x = torch.split(x, [self.p1, self.p2], dim=1)[self.id]

        return x
class SplitT_RGB(nn.Module):
    def __init__(self,p1,p2,id):
        super(Split, self).__init__()
        self.id = id
        self.p1 = p1
        self.p2 = p2
    def forward(self, x):
        # leftnum = x.shape[1]-3
        # return torch.split(x, [self.p1,self.p2], dim=1)[self.id]
        # x = torch.split(x, [self.p1, self.p2], dim=1)[self.id]
        x = torch.split(x, [self.p1, self.p2], dim=1)

        return x
class TimePart(nn.Module):
    def forward(self, x):
        # leftnum = x.shape[1]-3
        # return torch.split(x, [self.p1,self.p2], dim=1)[self.id]
        # x = torch.split(x, [self.p1, self.p2], dim=1)[self.id]
        x = x[0]

        return x
class PointPart(nn.Module):
    def forward(self, x):
        # leftnum = x.shape[1]-3
        # return torch.split(x, [self.p1,self.p2], dim=1)[self.id]
        # x = torch.split(x, [self.p1, self.p2], dim=1)[self.id]
        x = x[1]

        return x
class TimeMambaStem(nn.Module):
    def __init__(self, n_frames, out_channels,preconv=False):
        self.n_frames = n_frames
        super().__init__()
        # 第一阶段：1x1 混合帧间像素 (时间压缩)
        # self.time_pre_mix = Conv(n_frames*4, out_channels // 2, k=1)

        # 第二阶段：DwConv 提取空间特征
        self.spatial_dw1 = nn.Conv2d(n_frames, n_frames, kernel_size=3, stride=2, padding=1, groups=n_frames)
        self.spatial_dw2 = nn.Conv2d(n_frames, n_frames, kernel_size=3, stride=1, padding=1, groups=n_frames)
        self.spatial_dw3 = nn.Conv2d(n_frames, n_frames, kernel_size=3, stride=1, padding=1, groups=n_frames)
        self.spatial_dw4 = nn.Conv2d(n_frames, n_frames, kernel_size=3, stride=1, padding=1, groups=n_frames)

        # self.spatial_dw1 = nn.Sequential([nn.Conv2d(n_frames, n_frames, kernel_size=3, stride=2, padding=1, groups=n_frames) for _ in range(dep)])
        # self.spatial_dw2 = nn.Sequential([nn.Conv2d(n_frames, n_frames, kernel_size=3, stride=2, padding=1, groups=n_frames) for _ in range(dep)])
        # self.spatial_dw3 = nn.Sequential([nn.Conv2d(n_frames, n_frames, kernel_size=3, stride=2, padding=1, groups=n_frames) for _ in range(dep)])
        # self.spatial_dw4 = nn.Sequential([nn.Conv2d(n_frames, n_frames, kernel_size=3, stride=2, padding=1, groups=n_frames) for _ in range(dep)])
        # dep=4
        # self.spatial_dw1 = nn.Sequential([nn.Conv2d(n_frames, n_frames, kernel_size=3, stride=2, padding=1) for _ in range(dep)])
        # self.spatial_dw2 = nn.Sequential([nn.Conv2d(n_frames, n_frames, kernel_size=3, stride=2, padding=1) for _ in range(dep)])
        # self.spatial_dw3 = nn.Sequential([nn.Conv2d(n_frames, n_frames, kernel_size=3, stride=2, padding=1) for _ in range(dep)])
        # self.spatial_dw4 = nn.Sequential([nn.Conv2d(n_frames, n_frames, kernel_size=3, stride=2, padding=1) for _ in range(dep)])
        # 第三阶段：Mamba 建模长程依赖 (帧间混合增强)
        # 将 (B, C, H, W) 视为序列 (B, L, C)
        self.mamba = DisentangledSTMixSSM(inchannel=n_frames,preconv=preconv)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU()
        self.out_proj = Conv(4, out_channels, k=1)
        self.preconv = preconv
        # self.Preconv = nn.ModuleList(nn.Conv2d(in_channels=n_frames, out_channels=n_frames, kernel_size=3, stride=2, padding=1),
        #     nn.Conv2d(in_channels=n_frames, out_channels=n_frames*2, kernel_size=3, stride=1, padding=1),
        #                                *[nn.Conv2d(in_channels=n_frames*2, out_channels=n_frames*2, kernel_size=3, stride=1, padding=1) for _ in range(6)],
        #                         nn.Conv2d(in_channels=n_frames*2, out_channels=n_frames , kernel_size=3, stride=1, padding=1),
        # nn.Conv2d(in_channels=n_frames, out_channels=n_frames , kernel_size=3, stride=2, padding=1)
        #                              )
        # tc2 = n_frames*4
        # self.Preconv = nn.ModuleList([
        #     # Conv(c1=n_frames, c2=n_frames, k=3, s=2, p=1),
        #     Conv(c1=n_frames, c2=tc2, k=3, s=1, p=1),
        #
        #     # 核心修正：使用 * 解包列表推导式，消除嵌套层级
        #     *[Conv(c1=tc2, c2=tc2, k=3, s=1, p=1) for _
        #       in range(6)],
        #
        #     Conv(c1=tc2, c2=n_frames, k=3, s=1, p=1),
        #     Conv(c1=n_frames, c2=n_frames, k=3, s=2, p=1),
        #     Conv(c1=n_frames, c2=tc2, k=3, s=1, p=1),
        #     * [Conv(c1=tc2, c2=tc2, k=3, s=1, p=1) for _
        #        in range(6)],
        #     Conv(c1=tc2, c2=n_frames, k=3, s=1, p=1),
        #     # Conv(c1=n_frames, c2=n_frames, k=3, s=2, p=1),
        # ])
        self.alignment = DCNAlignment(4)

        # 保持 T4 原特征流形的 1x1 卷积 (为了与 DCN 处理过的特征在数值分布上对齐)
        self.t4_proj = nn.Conv2d(4, 4, kernel_size=1)
    def forward(self, x):
        # x = self.time_pre_mix(x)
        # x = self.time_pre_mix(x)+x
        # try:
        #     if self.preconv:
        #         for m in self.Preconv:
        #             x = m(x)
        # except:
        #     pass
        x1 = self.spatial_dw1(x)
        x2 = self.spatial_dw2(x1)
        x3 = self.spatial_dw3(x2)
        x4 = self.spatial_dw4(x3)
        t=[]
        for i in range(self.n_frames):
            t.append(torch.stack([x1[:, i], x2[:,i], x3[:,i], x4[:,i]],dim=1))

        # t2 = torch.stack([x1[:, 1], x2[:,1], x3[:,1], x4[:,1]],dim=1)
        # t3 = torch.stack([x1[:, 2], x2[:,2], x3[:,2], x4[:,2]],dim=1)
        # t4 = torch.stack([x1[:, 3], x2[:,3], x3[:,3], x4[:,3]],dim=1)
        #DCN
        try:
            for i in range(self.n_frames-1):
                t[i] = self.alignment(t[i],t[-1])
            t[-1] = self.t4_proj(t[-1])
        except:
            pass
        t = torch.cat(t, dim=1)
        # t = self.time_pre_mix(t)
        # Ma
        x = self.mamba(t)+t[:,-1:,:,:]
        x = self.out_proj(x)
        return self.act(self.bn(x))





class TimeMambaStemtransformer(nn.Module):
    """
    多尺度 Transformer 特征提取模块，兼容 YOLO 的 parse_model 参数传递方式。
    参数顺序必须与 YAML 中 args 列表一致，且要接收 c1, c2 两个自动传入的通道数。
    """

    def __init__(self, c1, d_model=256, nhead=8, num_layers=4, dim_feedforward=1024, dropout=0.1):
        """
        c1: 输入特征图的通道数（来自上层，可不用）
        c2: 输出通道数（YOLO 要求，这里实际上不使用，但需保留）
        d_model: Transformer 的 embedding 维度，也是投影后的通道数
        nhead: 多头注意力的头数，必须能整除 d_model
        """
        super().__init__()
        # 确保 d_model 为偶数（位置编码需要）
        if d_model % 2 != 0:
            d_model += 1
            print(f"Warning: d_model is odd, increased to {d_model} for positional encoding")

        # 确保 nhead 能整除 d_model
        while d_model % nhead != 0:
            nhead -= 1
            if nhead == 0:
                nhead = 1
                break
        print(f"MultiChannelTransformer: d_model={d_model}, nhead={nhead}, num_layers={num_layers}")

        self.d_model = d_model
        self.nhead = nhead

        # 输入投影（将任意输入通道映射到 d_model），将在 forward 中动态构建
        self.input_proj = None
        # 输出投影（将 d_model 映射回原始输入通道数），也在 forward 中动态构建
        self.output_proj = None

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, activation='gelu', batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, features):
        """
        features: list of [B, C_i, H_i, W_i]
        """
        features = [features]
        B = features[0].shape[0]
        device = features[0].device

        # 动态构建投影层（因为不同尺度的输入通道可能不同）
        if self.input_proj is None:
            self.input_proj = nn.ModuleList([
                nn.Conv2d(f.shape[1], self.d_model, kernel_size=1).to(device)
                for f in features
            ])
            self.output_proj = nn.ModuleList([
                nn.Conv2d(self.d_model, self.d_model, kernel_size=1,stride=2).to(device)
                for f in features
            ])

        tokens = []  # 存放所有尺度的 token 序列
        shapes = []  # 存放每个尺度的 (H, W)

        for i, feat in enumerate(features):
            # 1. 投影到 d_model 维度
            proj = self.input_proj[i](feat)  # [B, d_model, H, W]
            H, W = proj.shape[2], proj.shape[3]
            shapes.append((H, W))

            # 2. 展平为 token 序列 [B, L, d_model]
            tokens_i = proj.flatten(2).transpose(1, 2)  # [B, H*W, d_model]

            # 3. 添加 2D 位置编码（每个尺度独立）
            pos_enc = self._get_positional_encoding(H, W, self.d_model, device)  # [1, L, d_model]
            tokens_i = tokens_i + pos_enc
            tokens.append(tokens_i)

        # 4. 拼接所有尺度的 token
        tokens_concat = torch.cat(tokens, dim=1)  # [B, L_total, d_model]

        # 5. Transformer 全局交互
        tokens_enhanced = self.transformer(tokens_concat)  # [B, L_total, d_model]

        # 6. 分割回各尺度并重塑为特征图
        output_features = []
        start = 0
        for i, (H, W) in enumerate(shapes):
            L_i = H * W
            tokens_i = tokens_enhanced[:, start:start + L_i, :]  # [B, L_i, d_model]
            start += L_i
            # 重塑为特征图
            feat_i = tokens_i.transpose(1, 2).reshape(B, self.d_model, H, W)
            # 投影回原始通道数并加上残差连接
            feat_i = self.output_proj[i](feat_i)
            output_features.append(feat_i)

        return output_features[0]

    @staticmethod
    def _get_positional_encoding(H, W, d_model, device):
        """
        生成 2D 正弦余弦位置编码
        返回形状: [1, H*W, d_model]
        """
        assert d_model % 2 == 0, "d_model must be even"
        # 创建网格坐标
        y_coords = torch.arange(H, device=device).float()
        x_coords = torch.arange(W, device=device).float()
        # 归一化到 [0, 1] 范围（可选，也可以不归一化，但数值范围会影响频率）
        if H > 1:
            y_coords = y_coords / (H - 1)
        if W > 1:
            x_coords = x_coords / (W - 1)

        # 计算频率项
        dim_t = torch.arange(d_model // 2, device=device).float()
        dim_t = 10000 ** (2 * dim_t / d_model)  # 分母，形状 [d_model/2]

        # 计算位置编码：分别对 x 和 y 方向编码，然后相加
        # 形状处理：x_coords -> [1, W, 1] 以便广播
        x_embed = x_coords.unsqueeze(0).unsqueeze(-1)  # [1, W, 1]
        y_embed = y_coords.unsqueeze(1).unsqueeze(-1)  # [H, 1, 1]

        pos_x = x_embed * dim_t  # [1, W, d_model/2]
        pos_y = y_embed * dim_t  # [H, 1, d_model/2]

        # 正弦余弦交替
        pe_x = torch.stack([torch.sin(pos_x), torch.cos(pos_x)], dim=-1)  # [1, W, d_model/2, 2]
        pe_y = torch.stack([torch.sin(pos_y), torch.cos(pos_y)], dim=-1)  # [H, 1, d_model/2, 2]

        # 展平最后两维 -> [1, W, d_model] 和 [H, 1, d_model]
        pe_x = pe_x.flatten(-2)  # [1, W, d_model]
        pe_y = pe_y.flatten(-2)  # [H, 1, d_model]

        # 相加并展平空间维度: [H, W, d_model] -> [H*W, d_model]
        pe = (pe_y + pe_x).reshape(H * W, d_model)
        return pe.unsqueeze(0)  # [1, H*W, d_model]
# class LearnableFrequencyModulator(nn.Module):
#     """
#     轻量化可学习频域调制器 (Learnable Frequency Modulator)
#     集成特性:
#     1. 尺度不变性 (Scale Invariance): 采用双线性插值动态适配 YOLO 多尺度输入。
#     2. 混合精度安全 (AMP Safe): 强制 FFT 计算在 FP32 空间进行，防止特征消失。
#     """
#
#     def __init__(self, channels, base_h=128, base_w=128):
#         super().__init__()
#
#         self.channels = channels
#
#         # 物理限制: 实数快速傅里叶变换 (rfft2) 会根据 Nyquist 采样定理截断一半的冗余频率
#         self.base_freq_w = base_w // 2 + 1
#
#         # 频率先验掩膜 (Frequency Prior Mask)
#         # 维度: (通道数, 基础高度, 截断频率宽度, 实虚部)
#         # 采用实数定义以规避 PyTorch 优化器对 Complex64 参数的梯度更新 Bug
#         self.complex_weight = nn.Parameter(
#             torch.randn(channels, base_h, self.base_freq_w, 2, dtype=torch.float32) * 0.02
#         )
#
#         # 跨域特征融合 (Cross-domain Feature Fusion)
#         self.fusion_conv = nn.Conv2d(channels * 2, channels, kernel_size=1)
#         self.ssm = CCCBlock(channels*2,hidden_dim=8)
#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         """
#         Input x: (B, C, H, W) - 空间域实数张量
#         """
#         # --- 步骤 1: AMP 精度隔离与设备对齐 ---
#         original_dtype = x.dtype
#         device = x.device
#         # 强制转换为 FP32 执行频域计算，防止高频小信号 Underflow
#         x_float = x.to(torch.float32)
#         B, C, H, W = x_float.shape
#
#         # --- 步骤 2: 空间域 -> 频域变换 (RFFT2) ---
#         # rfft2 利用共轭对称性，显存占用仅为 fft2 的约 50%
#         # freq_x shape: (B, C, H, W // 2 + 1), dtype: complex64
#         freq_x = torch.fft.rfft2(x_float, norm='ortho')
#         _, _, H_freq, W_freq = freq_x.shape
#
#         # --- 步骤 3: 复数域向实数域映射 (Complex to Real Mapping) ---
#         # 将 complex64 拆分为 [实部, 虚部] 并压入通道维
#         # view_as_real 产生 shape: (B, C, H, W_freq, 2)
#         freq_real_view = torch.view_as_real(freq_x)
#
#         # 维度重排: (B, C, H, W_freq, 2) -> (B, 2, C, H, W_freq) -> (B, 2*C, H, W_freq)
#         freq_input = freq_real_view.permute(0, 4, 1, 2, 3).reshape(B, 2 * C, H, W_freq)
#         # 此时 freq_input 为纯实数 CUDA 张量，可直接喂入 CCCBlock
#
#         # --- 步骤 4: 频域闭环调制 (CCCBlock Processing) ---
#         # 警告：若 CCCBlock 内包含取模运算，务必确保其逻辑作用于实数域
#         # freq_modulated_real = self.ssm(freq_input)
#
#         # --- 步骤 5: 实数域还原为复数域 (Real to Complex Mapping) ---
#         # (B, 2*C, H, W_freq) -> (B, 2, C, H, W_freq) -> (B, C, H, W_freq, 2)
#         freq_modulated_split = freq_modulated_real.view(B, 2, C, H, W_freq).permute(0, 2, 3, 4, 1).contiguous()
#         # 重新封装为 complex64 类型
#         freq_modulated_complex = torch.view_as_complex(freq_modulated_split)
#
#         # --- 步骤 6: 频域 -> 空间域逆变换 (IRFFT2) ---
#         # 显式传入 s=(H, W) 以处理奇数宽度情况下的对齐
#         # spatial_from_freq shape: (B, C, H, W), dtype: float32
#         spatial_from_freq = torch.fft.irfft2(freq_modulated_complex, s=(H, W), norm='ortho')
#
#         # --- 步骤 7: 完美对齐与融合 (Fusion) ---
#         # 在通道维进行跳跃连接 (Skip Connection) 融合
#         fused_features = torch.cat([x_float, spatial_from_freq], dim=1)
#         out = self.fusion_conv(fused_features)
#
#         # --- 步骤 8: 恢复原始精度 ---
#         # 衔接下游 FP16/BF16 的 Conv 层，优化 4090 的计算吞吐
#         return out.to(original_dtype)
class LearnableFrequencyModulator(nn.Module):
    """
    轻量化可学习频域调制器 (基于 FFT 的空频闭环融合)
    """

    def __init__(self, channels, h, w):
        super(LearnableFrequencyModulator, self).__init__()

        # 对于实数 FFT (rfft2)，频域特征的宽度减半为 w//2 + 1
        # 确保输入图像尺寸固定，或采用自适应插值机制处理可变尺寸
        self.complex_weight = nn.Parameter(
            torch.randn(channels, h, w // 2 + 1, 2, dtype=torch.float32) * 0.02
        )

        # 融合融合空域和频域特征的 $1 \times 1$ 卷积
        self.fusion_conv = nn.Conv2d(channels * 2, channels, kernel_size=1)

    def forward(self, x):
        # 1. 保存空域残差副本
        spatial_residual = x

        # 2. 空域 -> 频域变换 (FFT)
        # 输入 shape: (B, C, H, W)，输出 shape: (B, C, H, W//2 + 1)
        # 注意: rfft2 默认输入为实数张量，输出为复数张量
        freq_x = torch.fft.rfft2(x, norm='ortho')

        # 3. 频域自适应调制
        # 将复数权重转换为 torch.complex64 并与频域特征逐元素相乘
        weight = torch.view_as_complex(self.complex_weight)
        freq_modulated = freq_x * weight

        # 4. 频域 -> 空域逆变换 (IFFT)
        # 逆变换后，输出恢复为空域张量，shape 回到 (B, C, H, W)
        spatial_from_freq = torch.fft.irfft2(freq_modulated, s=(x.size(-2), x.size(-1)), norm='ortho')

        # 5. 空频完美对齐与融合
        # 此时 spatial_from_freq 和 spatial_residual 像素一一对应
        fused_features = torch.cat([spatial_residual, spatial_from_freq], dim=1)
        out = self.fusion_conv(fused_features)

        return out
class DisentangledSTMixSSM(nn.Module):
    def __init__(
            self,
            inchannel: int = 0,
            preconv: bool = False,
            # outchannel: int = 0,
            hidden_dim: int = 4,#应等于此前dwconv个数
            drop_path: float = 0.1,
            channel_first=True,
            # =============================
            ssm_d_state: int = 16,
            ssm_ratio=2.0,#todo
            ssm_dt_rank: Any = "auto",
            ssm_act_layer=nn.SiLU,
            ssm_conv: int = 1,
            ssm_conv_bias=True,
            ssm_drop_rate: float = 0,
            ssm_init="v0",
            forward_type="v2",
            # =============================
            mlp_ratio=4.0, #4.0,
            mlp_act_layer=nn.GELU,
            mlp_drop_rate: float = 0.0,
            # =============================
            use_checkpoint: bool = False,
            post_norm: bool = False,
            # =============================
            **kwargs,
    ):
        super().__init__()
        self.inchannel = inchannel
        self.ssm_branch = ssm_ratio > 0
        self.mlp_branch = mlp_ratio > 0
        self.use_checkpoint = use_checkpoint
        self.post_norm = post_norm

        if self.ssm_branch:
            self.norm = LayerNorm(hidden_dim, channel_first=channel_first)
            self.op = SS2DTimeMix(#SS2D1conv,SS2D4conv,SS2Dv2
                inchannel=inchannel,
                d_model=hidden_dim,
                d_state=ssm_d_state,
                ssm_ratio=ssm_ratio,
                dt_rank=ssm_dt_rank,
                act_layer=ssm_act_layer,
                # ==========================
                d_conv=ssm_conv,
                conv_bias=ssm_conv_bias,
                # ==========================
                dropout=ssm_drop_rate,
                preconv=preconv,
                # bias=False,
                # ==========================
                # dt_min=0.001,
                # dt_max=0.1,
                # dt_init="random",
                # dt_scale="random",
                # dt_init_floor=1e-4,
                initialize=ssm_init,
                # ==========================
                forward_type=forward_type,
                channel_first=channel_first,
            )

        self.drop_path = DropPath(drop_path)
        # VSSBlock(
        #     hidden_dim=96,
        #     drop_path=0.1,
        #     channel_first=False,
        #     ssm_d_state=16,
        #     ssm_ratio=2.0,
        #     ssm_dt_rank="auto",
        #     ssm_act_layer=nn.SiLU,
        #     ssm_conv=3,
        #     ssm_conv_bias=True,
        #     ssm_drop_rate=0.0,
        #     ssm_init="v0",
        #     forward_type="v2",
        #     mlp_ratio=4.0,
        #     mlp_act_layer=nn.GELU,
        #     mlp_drop_rate=0.0,
        #     use_checkpoint=False,
        # )
        if self.mlp_branch:
            self.norm2 = LayerNorm(hidden_dim, channel_first=channel_first)
            mlp_hidden_dim = int(hidden_dim * mlp_ratio)
            self.mlp = Mlp(in_features=hidden_dim, hidden_features=mlp_hidden_dim,   act_layer=mlp_act_layer,#out_features=hidden_dim,
                           drop=mlp_drop_rate, channel_first=channel_first)
        # self.first_reshape = nn.Conv2d(4,hidden_dim,1,1)
        self.inchannel=inchannel
        # self.fft = LearnableFrequencyModulator(4,320,320)
    def _forward(self, input: torch.Tensor):
        xt = torch.chunk(input,chunks= self.inchannel,dim=1)
        # x = [self.norm(self.fft(xt[i])) for i in range(self.inchannel)]
        x = [self.norm(xt[i]) for i in range(self.inchannel)]
                # x = self.drop_path(self.op(self.norm(x)))
        x = x[-1] + self.drop_path(self.op(x))
        if self.mlp_branch:
            if self.post_norm:
                x = x + self.drop_path(self.norm2(self.mlp(x)))  # FFN
            else:
                x = x + self.drop_path(self.mlp(self.norm2(x)))  # FFN
                # x = self.drop_path(self.mlp(self.norm2(x)))
        return x
    def forward(self, input: torch.Tensor):
        if self.use_checkpoint:
            return checkpoint.checkpoint(self._forward, input)
        else:
            return self._forward(input)


class LearnableFFTLayerFull(nn.Module):
    def __init__(self, d, l):
        super(LearnableFFTLayerFull, self).__init__()
        self.l = l

        # 权重维度：(通道数 d, 序列全长 l)
        # 使用复数参数，直接调整幅度和相位
        self.complex_weight = nn.Parameter(
            torch.randn(d, l, dtype=torch.complex64) * 0.02
        )

    def forward(self, x):
        # x 维度: (b, d, l)

        # 1. 全长度快速傅里叶变换
        # 输出维度: (b, d, l), 类型: complex64
        x_freq = torch.fft.fft(x, dim=-1)

        # 2. 频域点乘
        # (b, d, l) * (d, l) -> (b, d, l)
        x_freq = x_freq * self.complex_weight

        # 3. 逆傅里叶变换
        # 输出维度: (b, d, l), 类型: complex64
        x_res = torch.fft.ifft(x_freq, dim=-1)

        # 4. 取实部回到原尺寸实数空间
        # 注意：如果输入本身是复数，则不应调用 .real
        return torch.real(x_res)
class SS2DTimeMix(nn.Module):
    def __init__(
            self,
            # basic dims ===========
            inchannel,
            preconv,
            d_model=96,
            d_state=16,
            ssm_ratio=2.0,
            dt_rank="auto",
            act_layer=nn.SiLU,
            # dwconv ===============
            d_conv=1,  # < 2 means no conv
            conv_bias=True,
            # ======================
            dropout=0.0,
            bias=False,
            # dt init ==============
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            initialize="v0",
            # ======================
            forward_type="v2",
            channel_first=False,
            # ======================
            **kwargs,
            # VSSBlock(
            # hidden_dim=96,
            # drop_path=0.1,
            # channel_first=False,
            # ssm_d_state=16,
            # ssm_ratio=2.0,
            # ssm_dt_rank="auto",
            # ssm_act_layer=nn.SiLU,
            # ssm_conv=3,
            # ssm_conv_bias=True,
            # ssm_drop_rate=0.0,
            # ssm_init="v0",
            # forward_type="v2",
            # mlp_ratio=4.0,
            # mlp_act_layer=nn.GELU,
            # mlp_drop_rate=0.0,
            # use_checkpoint=False,
            # )
    ):
        factory_kwargs = {"device": None, "dtype": None}
        super().__init__()
        self.inchannel = inchannel
        self.k_group = 1
        self.d_model = int(d_model)
        self.d_state = int(d_state)
        self.d_inner = int(ssm_ratio * d_model)
        self.dt_rank = int(math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank)
        self.channel_first = channel_first
        self.with_dconv = d_conv > 1
        # self.forward = self.forward

        # todo tags for forward_type =============================="ori"
        # checkpostfix = self.checkpostfix
        # self.disable_force32, forward_type = checkpostfix("_no32", forward_type)
        # self.oact, forward_type = checkpostfix("_oact", forward_type)
        # self.disable_z, forward_type = checkpostfix("_noz", forward_type)
        # self.disable_z_act, forward_type = checkpostfix("_nozact", forward_type)
        # self.out_norm, forward_type = self.get_outnorm(forward_type, self.d_inner, channel_first)
        #todo ====================================================1scan conv concat
        self.disable_force32 = False
        self.oact = False
        self.disable_z = True
        self.disable_z_act = True
        self.out_norm , forward_type= self.get_outnorm(forward_type, self.d_inner, channel_first)
        # forward_type debug =======================================
        FORWARD_TYPES = dict(
            # v01=partial(self.forward_corev2, force_fp32=(not self.disable_force32), selective_scan_backend="mamba",
            #             scan_force_torch=True),
            # v02=partial(self.forward_corev2, force_fp32=(not self.disable_force32), selective_scan_backend="mamba"),
            # v03=partial(self.forward_corev2, force_fp32=(not self.disable_force32), selective_scan_backend="oflex"),
            # v04=partial(self.forward_corev2, force_fp32=False),  # selective_scan_backend="oflex", scan_mode="cross2d"
            # v05=partial(self.forward_corev2, force_fp32=False, no_einsum=True),
            # # selective_scan_backend="oflex", scan_mode="cross2d"
            # # ===============================
            # v051d=partial(self.forward_corev2, force_fp32=False, no_einsum=True, scan_mode="unidi"),
            # v052d=partial(self.forward_corev2, force_fp32=False, no_einsum=True, scan_mode="bidi"),
            # v052dc=partial(self.forward_corev2, force_fp32=False, no_einsum=True, scan_mode="cascade2d"),
            # v052d3=partial(self.forward_corev2, force_fp32=False, no_einsum=True, scan_mode=3),  # debug
            # ===============================
            v2=partial(self.forward_corev2, force_fp32=(not self.disable_force32), selective_scan_backend="core"),
            # v3=partial(self.forward_corev2, force_fp32=False, selective_scan_backend="oflex"),
        )
        self.forward_core = FORWARD_TYPES.get(forward_type, None)

        # in proj =======================================
        d_proj = self.d_inner if self.disable_z else (self.d_inner * 2)
        self.in_proj = Linear(d_model, d_proj, bias=bias, channel_first=channel_first)
        self.act: nn.Module = act_layer()

        # conv =======================================
        if self.with_dconv:
            self.conv2d = nn.Conv2d(
                in_channels=self.d_inner,
                out_channels=self.d_inner,
                groups=self.d_inner,
                bias=conv_bias,
                kernel_size=d_conv,
                padding=(d_conv - 1) // 2,
                **factory_kwargs,
            )

        # x proj ============================
        #todo
        self.x_proj = [nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False)]
        # self.x_proj = Linear(self.d_inner, self.k_group * (self.dt_rank + self.d_state * 2), groups=self.k_group,
        #                      bias=False, channel_first=True)
        self.dt_projs = Linear(self.dt_rank, self.k_group * self.d_inner, groups=self.k_group, bias=False,
                               channel_first=True)

        # self.x_proj = [
        #     nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False)
        #     for _ in range(self.k_group)
        # ]
        # self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0)) # (K, N, inner)
        # del self.x_proj

        # out proj =======================================
        self.out_act = nn.GELU() if self.oact else nn.Identity()
        self.out_proj = Linear(self.d_inner, self.d_model, bias=bias, channel_first=channel_first)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else nn.Identity()

        if initialize in ["v0"]:
            self.A_logs, self.Ds, self.dt_projs_weight, self.dt_projs_bias = mamba_init.init_dt_A_D(
                self.d_state, self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                k_group=self.k_group,
            )
        elif initialize in ["v1"]:
            # simple init dt_projs, A_logs, Ds
            self.Ds = nn.Parameter(torch.ones((self.k_group * self.d_inner)))
            self.A_logs = nn.Parameter(torch.randn(
                (self.k_group * self.d_inner, self.d_state)))  # A == -A_logs.exp() < 0; # 0 < exp(A * dt) < 1
            self.dt_projs_weight = nn.Parameter(
                0.1 * torch.randn((self.k_group, self.d_inner, self.dt_rank)))  # 0.1 is added in 0430
            self.dt_projs_bias = nn.Parameter(0.1 * torch.randn((self.k_group, self.d_inner)))  # 0.1 is added in 0430
        elif initialize in ["v2"]:
            # simple init dt_projs, A_logs, Ds
            self.Ds = nn.Parameter(torch.ones((self.k_group * self.d_inner)))
            self.A_logs = nn.Parameter(torch.zeros(
                (self.k_group * self.d_inner, self.d_state)))  # A == -A_logs.exp() < 0; # 0 < exp(A * dt) < 1

            self.dt_projs_weight = nn.Parameter(0.1 * torch.rand((self.k_group, self.d_inner, self.dt_rank)))
            self.dt_projs_bias = nn.Parameter(0.1 * torch.rand((self.k_group, self.d_inner)))
        self.dt_projs.weight.data = self.dt_projs_weight.data.view(self.dt_projs.weight.shape)
        # self.dt_projs.bias.data = self.dt_projs_bias.data.view(self.dt_projs.bias.shape)
        del self.dt_projs_weight
        # del self.dt_projs_bias
        #todo

        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0))
        # if not preconv:
        #     self.fft = LearnableFFTLayerFull(self.d_inner,512000)
        #     # self.fft = LearnableFFTLayerFull(self.d_inner, 921600)
        # else:
        #     self.fft = LearnableFFTLayerFull(self.d_inner, 128000)

    def forward_corev2(
            self,
            x: torch.Tensor = None,
            # ==============================
            force_fp32=False,  # True: input fp32
            # ==============================
            ssoflex=True,  # True: input 16 or 32 output 32 False: output dtype as input
            # ==============================
            selective_scan_backend=None,#v2
            # ==============================
            scan_mode="unidi",#cross2d=0, unidi=1, bidi=2, cascade2d=-1
            scan_force_torch=False,
            # ==============================
            **kwargs,
    ):
        # assert selective_scan_backend in [None, "oflex", "mamba", "torch"]
        _scan_mode = dict(cross2d=0, unidi=1, bidi=2, cascade2d=-1).get(scan_mode, None) if isinstance(scan_mode,
                                                                                                       str) else scan_mode  # for debug
        assert isinstance(_scan_mode, int)
        delta_softplus = True
        channel_first = self.channel_first
        to_fp32 = lambda *args: (_a.to(torch.float32) for _a in args)
        force_fp32 = force_fp32 or ((not ssoflex) and self.training)


        B, D, H, W = x[0].shape
        N = self.d_state
        K, D, R = self.k_group, self.d_inner, self.dt_rank
        L = H * W


        def selective_scan(u, delta, A, B, C, D=None, delta_bias=None, delta_softplus=True):
            return selective_scan_fn(u, delta, A, B, C, D, delta_bias, delta_softplus, ssoflex,
                                     backend=selective_scan_backend)

        if True:
            xst = [cross_scan_fn(i, in_channel_first=True, out_channel_first=True, scans=_scan_mode,
                               force_torch=scan_force_torch) for i in x]
            #todo
            xs = x[0].new_empty(B, self.d_inner, L, self.inchannel)

            for j in range(self.inchannel):
                xs[:, :, :, j] = xst[j][:,0]
            xs = xs.flatten(2, 3)

            # xs = self.fft(xs)
            txs = x[0].new_empty((B, 1, self.d_inner, self.inchannel * L))
            txs[:, 0] = xs
            # x_con = self.my_conv(x)
            # # x_con = conv_norm(x_con)
            # x_con = x_con.flatten(2, 3)
            # # x_cons = x.new_empty((B, K,  D+N+N, H * W))#tdo BC
            # x_cons = x.new_empty((B, self.d_inner, H * W, 2))
            # x_cons[:, :, :, 0] = xs[:, 0, :, :]
            # x_cons[:, :, :, 1] = x_con
            # x_conmulty = x_cons.flatten(2, 3)
            # x_consmulty = x.new_empty((B, 1, self.d_inner, 2 * H * W))
            # x_consmulty[:, 0] = x_conmulty

            x_dbl = torch.einsum("b k d l, k c d -> b k c l", txs, self.x_proj_weight)  # 批，方向，矩阵相乘。proj矩阵预测Δ，B，C

            #todo
            # x_dbl = self.x_proj(xs.view(B, -1, L))
            # dts, Bs, Cs = torch.split(x_dbl.contiguous().view(B, K, -1, L), [R, N, N], dim=2)
            dts, Bs, Cs = torch.split(x_dbl, [R, N, N], dim=2)
            dts = dts.contiguous().view(B, -1, L*self.inchannel)
            dts = self.dt_projs(dts)
            # dts = self.fft(dts)
            xs = xs.view(B, -1, L*self.inchannel)
            dts = dts.contiguous().view(B, -1, L*self.inchannel)
            As = -self.A_logs.to(torch.float).exp()  # (k * c, d_state)
            Ds = self.Ds.to(torch.float)  # (K * c)
            # Bs = self.fft(Bs)
            Bs = Bs.contiguous().view(B, K, N, L*self.inchannel)
            Cs = Cs.contiguous().view(B, K, N, L*self.inchannel)
            delta_bias = self.dt_projs_bias.view(-1).to(torch.float)

            if force_fp32:
                xs, dts, Bs, Cs = to_fp32(xs, dts, Bs, Cs)

            yst: torch.Tensor = selective_scan(
                xs, dts, As, Bs, Cs, Ds, delta_bias, delta_softplus
            )
            # todo
            ys = yst.view(B, -1, L, self.inchannel).sum(dim=3)
            # for ti in range(self.inchannel):
            #     ys+=yst[:, :, ti::self.inchannel]
            ys=ys.contiguous().view(B, K, -1, H, W)
            # ys = (ys[:, :, ::self.inchannel] + ys[:, :, 1::self.inchannel]).contiguous().view(B, K, -1, H, W)
            # ys = ys[:, :, ::2].contiguous().view(B, K, -1, H, W)
            y: torch.Tensor = cross_merge_fn(ys, in_channel_first=True, out_channel_first=True, scans=_scan_mode,
                                             force_torch=scan_force_torch)

            if getattr(self, "__DEBUG__", False):
                setattr(self, "__data__", dict(
                    A_logs=self.A_logs, Bs=Bs, Cs=Cs, Ds=Ds,
                    us=xs, dts=dts, delta_bias=delta_bias,
                    ys=ys, y=y, H=H, W=W,
                ))

        y = y.view(B, -1, H, W)
        if not channel_first:
            y = y.permute(0, 2, 3, 1).contiguous()

        y = self.out_norm(y)

        return y.to(x[0].dtype)

    def forward(self, x: torch.Tensor, **kwargs):

        x = [self.in_proj(x[i]) for i in range(self.inchannel)]
        # x = self.in_proj(x)
        # if not self.disable_z:
        #     x, z = x.chunk(2, dim=(1 if self.channel_first else -1))  # (b, h, w, d)
        #     if not self.disable_z_act:
        #         z = self.act(z)
        if not self.channel_first:
            x = x.permute(0, 3, 1, 2).contiguous()
        # if self.with_dconv:
        #     x = self.conv2d(x)  # (b, d, h, w)
        # x = self.act(x)
        x = [self.act(x[i]) for i in range(self.inchannel)]
        y = self.forward_core(x)
        y = self.out_act(y)
        # if not self.disable_z:
        #     y = y * z
        out = self.dropout(self.out_proj(y))
        return out

    @staticmethod
    def get_outnorm(forward_type="", d_inner=192, channel_first=True):
        def checkpostfix(tag, value):
            ret = value[-len(tag):] == tag
            if ret:
                value = value[:-len(tag)]
            return ret, value

        out_norm_none, forward_type = checkpostfix("_onnone", forward_type)
        out_norm_dwconv3, forward_type = checkpostfix("_ondwconv3", forward_type)
        out_norm_cnorm, forward_type = checkpostfix("_oncnorm", forward_type)
        out_norm_softmax, forward_type = checkpostfix("_onsoftmax", forward_type)
        out_norm_sigmoid, forward_type = checkpostfix("_onsigmoid", forward_type)

        out_norm = nn.Identity()
        if out_norm_none:
            out_norm = nn.Identity()
        elif out_norm_cnorm:
            out_norm = nn.Sequential(
                LayerNorm(d_inner, channel_first=channel_first),
                (nn.Identity() if channel_first else Permute(0, 3, 1, 2)),
                nn.Conv2d(d_inner, d_inner, kernel_size=3, padding=1, groups=d_inner, bias=False),
                (nn.Identity() if channel_first else Permute(0, 2, 3, 1)),
            )
        elif out_norm_dwconv3:
            out_norm = nn.Sequential(
                (nn.Identity() if channel_first else Permute(0, 3, 1, 2)),
                nn.Conv2d(d_inner, d_inner, kernel_size=3, padding=1, groups=d_inner, bias=False),
                (nn.Identity() if channel_first else Permute(0, 2, 3, 1)),
            )
        elif out_norm_softmax:
            out_norm = SoftmaxSpatial(dim=(-1 if channel_first else 1))
        elif out_norm_sigmoid:
            out_norm = nn.Sigmoid()
        else:
            out_norm = LayerNorm(d_inner, channel_first=channel_first)

        return out_norm, forward_type

    @staticmethod
    def checkpostfix(tag, value):
        ret = value[-len(tag):] == tag
        if ret:
            value = value[:-len(tag)]
        return ret, value
import torchvision.ops as ops

class DCNAlignment(nn.Module):
    """
    单帧对齐模块：使用当前帧 T_key 引导历史帧 T_ref 进行空间对齐
    """

    def __init__(self, channels, kernel_size=3):
        super(DCNAlignment, self).__init__()
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2

        # 偏移量预测网络：输入为拼接的特征 (2*C)，输出为 Offset 和 Mask (3 * K^2)
        # K^2 个空间位置，每个位置需要 \Delta x, \Delta y (共 2 个) 以及 1 个权重 mask
        self.conv_offset = nn.Conv2d(
            in_channels=channels * 2,
            out_channels=3 * kernel_size * kernel_size,
            kernel_size=3,
            padding=1,
            bias=True
        )
        # self.fft = LearnableFrequencyModulator(self.kernel_size ** 2,640,640)
        # self.conv_offset = CCCBlock(channels*2,3 * kernel_size * kernel_size)
        # mid_channels = channels *4
        # self.conv_offset = nn.Sequential(
        #     # 特征融合与升维
        #     Conv(c1=channels * 2, c2=mid_channels, k=3, s=1, p=1),
        #
        #     # 使用列表推导式构建深层处理 (推荐 2-3 层即可，无需 6 层)
        #     # 为了扩大感受野而不增加计算量，可以引入空洞卷积 (d=2)
        #     *[Conv(c1=mid_channels, c2=mid_channels, k=3, s=1, p=2, d=2) for _ in range(2)],
        #
        #     # 降维回基础特征厚度
        #     Conv(c1=mid_channels, c2=channels, k=3, s=1, p=1),
        #
        #     # 【核心致命错误修正】: 最后一层必须是纯粹的 nn.Conv2d
        #     # - 输出通道必须绝对等于 3 * K^2
        #     # - 绝对不能带有 BN 和 SiLU (保证偏移量能输出负数，掩码不受干扰)
        #     nn.Conv2d(
        #         in_channels=channels,
        #         out_channels=3 * kernel_size * kernel_size, # 3x3卷积此处为27
        #         kernel_size=3,
        #         padding=1,
        #         bias=True
        #     ))


        # 原生的可变形卷积层 (DCN v2)
        self.dcn = ops.DeformConv2d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=kernel_size,
            padding=self.padding,
            bias=True
        )

        # self._init_weights()

    def _init_weights(self):
        # 【极其重要】初始化 offset 卷积的权重和偏置为 0
        # 这确保了网络在初始状态下，偏移量为 0，DCN 退化为标准的 3x3 卷积
        # 避免模型在训练初期因为随机的剧烈位移而直接崩溃
        nn.init.constant_(self.conv_offset.weight, 0)
        nn.init.constant_(self.conv_offset.bias, 0)
        # nn.init.constant_(self.conv_offset[-1].weight, 0)
        # nn.init.constant_(self.conv_offset[-1].bias, 0)

    def forward(self, t_ref, t_key):
        """
        t_ref: 需要被变形的历史帧特征 (B, C, H, W)
        t_key: 作为锚点的当前帧特征 (B, C, H, W)
        """
        # 1. 拼接特征，感知运动差异
        concat_feat = torch.cat([t_ref, t_key], dim=1)  # (B, 2C, H, W)

        # 2. 预测偏移场和掩码
        offset_mask = self.conv_offset(concat_feat)

        # 拆分出 offset (前 2*K^2 个通道) 和 mask (最后 K^2 个通道)
        o1, o2 = 2 * self.kernel_size ** 2, 3 * self.kernel_size ** 2
        offset = offset_mask[:, :o1, :, :]
        mask = torch.sigmoid(offset_mask[:, o1:o2, :, :])  # Mask 必须在 0~1 之间
        # mask = torch.sigmoid(self.fft(offset_mask[:, o1:o2, :, :]))
        # 3. 执行变形对齐 (Warping)
        # 注意：是对 t_ref 进行变形，抽取其特征对齐到 t_key 的空间位置
        aligned_feat = self.dcn(t_ref, offset,  mask=mask)
        return aligned_feat


class STFusionWithAlignment(nn.Module):
    """
    完整的时空融合前置模块：包含对齐 + 交错扫描
    """

    def __init__(self, channels):
        super(STFusionWithAlignment, self).__init__()
        # 共享的对齐模块 (或者你可以为 T1, T2, T3 分别实例化三个独立的对齐模块)
        self.alignment = DCNAlignment(channels)

        # 保持 T4 原特征流形的 1x1 卷积 (为了与 DCN 处理过的特征在数值分布上对齐)
        self.t4_proj = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, t1, t2, t3, t4):
        """
        输入均为 (B, C, H, W)
        """
        # 1. 历史帧强制对齐到当前帧 T4 (消除背景运动位移)
        t1_aligned = self.alignment(t1, t4)
        t2_aligned = self.alignment(t2, t4)
        t3_aligned = self.alignment(t3, t4)
        t4_proj = self.t4_proj(t4)

        # 2. 堆叠特征
        # stack_feat shape: (B, 4, C, H, W)
        stack_feat = torch.stack([t1_aligned, t2_aligned, t3_aligned, t4_proj], dim=1)

        B, T, C, H, W = stack_feat.shape

        # 3. 执行 Interleaving Scan (交错扫描: 1111 2222 3333...)
        # 目前的空间顺序是: 先按通道，再按时间，再按空间。
        # 我们需要把它变换为：同一空间位置 (h, w) 下，连续排列 4 个时间步。

        # 第1步: 维度重排 -> (B, C, H, W, T)
        interleaved_feat = stack_feat.permute(0, 2, 3, 4, 1)

        # 第2步: 展平时间与空间维度 -> (B, C, H*W*T)
        # 此时的序列物理意义就是： [h0w0_t1, h0w0_t2, h0w0_t3, h0w0_t4, h0w1_t1, h0w1_t2...]
        # 完美对应图中的 1 1 1 1 2 2 2 2 ... 序列
        ssm_input_sequence = interleaved_feat.reshape(B, C, -1)

        # 后续可以直接送入 SSM 模块...
        return ssm_input_sequence
class TMixSSM(nn.Module):
    def __init__(
            self,
            inchannel: [int] ,
            # outchannel: int = 0,
            hidden_dim: int = 128,#应等于此前dwconv个数
            drop_path: float = 0.1,
            channel_first=True,
            # =============================
            ssm_d_state: int = 16,
            ssm_ratio=2.0,#todo
            ssm_dt_rank: Any = "auto",
            ssm_act_layer=nn.SiLU,
            ssm_conv: int = 1,
            ssm_conv_bias=True,
            ssm_drop_rate: float = 0,
            ssm_init="v0",
            forward_type="v2",
            # =============================
            mlp_ratio=4.0,
            mlp_act_layer=nn.GELU,
            mlp_drop_rate: float = 0.0,
            # =============================
            use_checkpoint: bool = False,
            post_norm: bool = False,
            # =============================
            **kwargs,
    ):
        super().__init__()
        # self.inchannel = inchannel
        self.ssm_branch = ssm_ratio > 0
        self.mlp_branch = mlp_ratio > 0
        self.use_checkpoint = use_checkpoint
        self.post_norm = post_norm

        if self.ssm_branch:
            self.norm = LayerNorm(hidden_dim, channel_first=channel_first)
            self.op = SS2DguideMix(#SS2D1conv,SS2D4conv,SS2Dv2
                # inchannel=inchannel,
                d_model=hidden_dim,
                d_state=ssm_d_state,
                ssm_ratio=ssm_ratio,
                dt_rank=ssm_dt_rank,
                act_layer=ssm_act_layer,
                # ==========================
                d_conv=ssm_conv,
                conv_bias=ssm_conv_bias,
                # ==========================
                dropout=ssm_drop_rate,
                # bias=False,
                # ==========================
                # dt_min=0.001,
                # dt_max=0.1,
                # dt_init="random",
                # dt_scale="random",
                # dt_init_floor=1e-4,
                initialize=ssm_init,
                # ==========================
                forward_type=forward_type,
                channel_first=channel_first,
            )

        self.drop_path = DropPath(drop_path)
        # VSSBlock(
        #     hidden_dim=96,
        #     drop_path=0.1,
        #     channel_first=False,
        #     ssm_d_state=16,
        #     ssm_ratio=2.0,
        #     ssm_dt_rank="auto",
        #     ssm_act_layer=nn.SiLU,
        #     ssm_conv=3,
        #     ssm_conv_bias=True,
        #     ssm_drop_rate=0.0,
        #     ssm_init="v0",
        #     forward_type="v2",
        #     mlp_ratio=4.0,
        #     mlp_act_layer=nn.GELU,
        #     mlp_drop_rate=0.0,
        #     use_checkpoint=False,
        # )
        if self.mlp_branch:
            self.norm2 = LayerNorm(hidden_dim, channel_first=channel_first)
            mlp_hidden_dim = int(hidden_dim * mlp_ratio)
            self.mlp = Mlp(in_features=hidden_dim, hidden_features=mlp_hidden_dim,   act_layer=mlp_act_layer,#out_features=hidden_dim,
                           drop=mlp_drop_rate, channel_first=channel_first)
        # self.first_reshape = nn.Conv2d(4,hidden_dim,1,1)
        # self.fft1 = LearnableFrequencyModulator(inchannel[0], 640, 640)
        # self.fft2 = LearnableFrequencyModulator(inchannel[1], 640, 640)
        self.saam = TaskAlignedPredictor(inchannel[1],inchannel[1])
        self.haar=HaarWaveletDownsampling(inchannel[1], inchannel[0], scale_factor=int(inchannel[0]/inchannel[1]))#scale_factor
    def _forward(self, input: [torch.Tensor]):
        # xt = torch.chunk(input,chunks= self.inchannel,dim=1)
        x0 = input[0]
        # x0 = self.fft1(input[0])

        x1 = input[1]
        # x1 = self.fft2(x1)
        # x1 = self.saam(x1)
        x1 = self.haar(x1)

        x = [x0, x1]
        x = [self.norm(i) for i in x]
                # x = self.drop_path(self.op(self.norm(x)))
        x = x[0] + self.drop_path(self.op(x))
        if self.mlp_branch:
            if self.post_norm:
                x = x + self.drop_path(self.norm2(self.mlp(x)))  # FFN
            else:
                x = x + self.drop_path(self.mlp(self.norm2(x)))  # FFN
                # x = self.drop_path(self.mlp(self.norm2(x)))
        return x
    def forward(self, input: [torch.Tensor]):
        if self.use_checkpoint:
            return checkpoint.checkpoint(self._forward, input)
        else:
            return self._forward(input)


class TaskAlignedPredictor(nn.Module):
    """
    任务对齐预测器 (Task-Aligned Predictor, TAP)
    用于在分类和回归分支解耦前，计算对齐特征与空间注意力权重，纠正 SSM 带来的特征漂移。
    """

    def __init__(self, in_channels, hidden_channels):
        super().__init__()
        # 特征降维与交互准备
        self.interact_conv = Conv(in_channels, hidden_channels, 3)

        # 预测空间对齐图 (Spatial Alignment Map)
        # 输出单通道特征图，表示每个空间像素点上分类与回归任务的对齐置信度
        self.alignment_map = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels // 2, 1),
            nn.BatchNorm2d(hidden_channels // 2),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden_channels // 2, 1, 1),
            nn.Sigmoid()  # 将对齐权重归一化到 (0, 1)
        )

    def forward(self, x):
        # 提取交互特征
        interact_feat = self.interact_conv(x)
        # 计算空间对齐权重
        align_weight = self.alignment_map(interact_feat)
        # 利用注意力机制重新校准特征，抑制发生空间漂移的劣质特征点
        aligned_feat = interact_feat * align_weight
        return aligned_feat
#channel wise
class HaarWaveletDownsamplingSize(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor=8):
        super().__init__()
        self.scale_factor = scale_factor
        self.levels = int((math.log2(scale_factor))/2)
        poolsize = 2 ** self.levels
        # poolsize = 2 * 2 ** self.levels
        self.pool = nn.AvgPool2d(kernel_size=poolsize, stride=poolsize, padding=0)
        # final_dim = in_channels * (4 ** self.levels)
        # self.channel_reducer = nn.Conv2d(final_dim, out_channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.register_buffer('haar_weights', self._create_haar_kernel(in_channels))
    def _create_haar_kernel(self, channels):
        weights = torch.zeros(channels * 4, channels, 2, 2)
        for c in range(channels):
            # LL (Low-Low)
            weights[c * 4 + 0, c, :, :] = 0.5 * torch.tensor([[1, 1], [1, 1]])
            # LH (Low-High) - Vertical Edges
            weights[c * 4 + 1, c, :, :] = 0.5 * torch.tensor([[-1, -1], [1, 1]])
            # HL (High-Low) - Horizontal Edges
            weights[c * 4 + 2, c, :, :] = 0.5 * torch.tensor([[-1, 1], [-1, 1]])
            # HH (High-High) - Diagonal Edges
            weights[c * 4 + 3, c, :, :] = 0.5 * torch.tensor([[1, -1], [-1, 1]])
        return weights
    def dwt_step(self, x, current_channels):
        b, c, h, w = x.shape
        x_reshaped = x.view(b, c, h // 2, 2, w // 2, 2)
        x_00 = x_reshaped[:, :, :, 0, :, 0]

        # 提取右上 (Row=0, Col=1)
        x_01 = x_reshaped[:, :, :, 0, :, 1]

        # 提取左下 (Row=1, Col=0)
        x_10 = x_reshaped[:, :, :, 1, :, 0]

        # 提取右下 (Row=1, Col=1)
        x_11 = x_reshaped[:, :, :, 1, :, 1]

        # Haar 变换公式
        ll = (x_00 + x_01 + x_10 + x_11) / 2
        lh = (x_00 + x_01 - x_10 - x_11) / 2
        hl = (x_00 - x_01 + x_10 - x_11) / 2
        hh = (x_00 - x_01 - x_10 + x_11) / 2

        return torch.cat([ll, lh, hl, hh], dim=1)

    def forward(self, x):
        out = x
        curr_c = x.shape[1]

        # 递归执行 DWT (3次)
        for _ in range(self.levels):
            out = self.dwt_step(out, curr_c)
            curr_c *= 4

            # 此时 out shape: [1, 3072, 16, 16]
        # 降维对齐
        # out = self.channel_reducer(out)
        out = self.pool(out)

        out = self.bn(out)
        return out

class HaarWaveletDownsampling(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor=8):
        super().__init__()
        self.scale_factor = scale_factor
        self.levels = int(math.log2(scale_factor))
        final_dim = in_channels * (4 ** self.levels)
        self.channel_reducer = nn.Conv2d(final_dim, out_channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.register_buffer('haar_weights', self._create_haar_kernel(in_channels))
    def _create_haar_kernel(self, channels):
        weights = torch.zeros(channels * 4, channels, 2, 2)
        for c in range(channels):
            # LL (Low-Low)
            weights[c * 4 + 0, c, :, :] = 0.5 * torch.tensor([[1, 1], [1, 1]])
            # LH (Low-High) - Vertical Edges
            weights[c * 4 + 1, c, :, :] = 0.5 * torch.tensor([[-1, -1], [1, 1]])
            # HL (High-Low) - Horizontal Edges
            weights[c * 4 + 2, c, :, :] = 0.5 * torch.tensor([[-1, 1], [-1, 1]])
            # HH (High-High) - Diagonal Edges
            weights[c * 4 + 3, c, :, :] = 0.5 * torch.tensor([[1, -1], [-1, 1]])
        return weights
    def dwt_step(self, x, current_channels):
        b, c, h, w = x.shape
        x_reshaped = x.view(b, c, h // 2, 2, w // 2, 2)
        x_00 = x_reshaped[:, :, :, 0, :, 0]

        # 提取右上 (Row=0, Col=1)
        x_01 = x_reshaped[:, :, :, 0, :, 1]

        # 提取左下 (Row=1, Col=0)
        x_10 = x_reshaped[:, :, :, 1, :, 0]

        # 提取右下 (Row=1, Col=1)
        x_11 = x_reshaped[:, :, :, 1, :, 1]

        # Haar 变换公式
        ll = (x_00 + x_01 + x_10 + x_11) / 2
        lh = (x_00 + x_01 - x_10 - x_11) / 2
        hl = (x_00 - x_01 + x_10 - x_11) / 2
        hh = (x_00 - x_01 - x_10 + x_11) / 2

        return torch.cat([ll, lh, hl, hh], dim=1)

    def forward(self, x):
        out = x
        curr_c = x.shape[1]

        # 递归执行 DWT (3次)
        for _ in range(self.levels):
            out = self.dwt_step(out, curr_c)
            curr_c *= 4

            # 此时 out shape: [1, 3072, 16, 16]
        # 降维对齐
        out = self.channel_reducer(out)
        out = self.bn(out)
        return out

class SS2DguideMix(nn.Module):
    def __init__(
            self,
            # basic dims ===========
            # inchannel,
            d_model=96,
            d_state=16,
            ssm_ratio=2.0,
            dt_rank="auto",
            act_layer=nn.SiLU,
            # dwconv ===============
            d_conv=1,  # < 2 means no conv
            conv_bias=True,
            # ======================
            dropout=0.0,
            bias=False,
            # dt init ==============
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            initialize="v0",
            # ======================
            forward_type="v2",
            channel_first=False,
            # ======================
            **kwargs,
            # VSSBlock(
            # hidden_dim=96,
            # drop_path=0.1,
            # channel_first=False,
            # ssm_d_state=16,
            # ssm_ratio=2.0,
            # ssm_dt_rank="auto",
            # ssm_act_layer=nn.SiLU,
            # ssm_conv=3,
            # ssm_conv_bias=True,
            # ssm_drop_rate=0.0,
            # ssm_init="v0",
            # forward_type="v2",
            # mlp_ratio=4.0,
            # mlp_act_layer=nn.GELU,
            # mlp_drop_rate=0.0,
            # use_checkpoint=False,
            # )
    ):
        factory_kwargs = {"device": None, "dtype": None}
        super().__init__()
        # self.inchannel = inchannel
        self.inchannel = 2
        self.k_group = 1
        self.d_model = int(d_model)
        self.d_state = int(d_state)
        self.d_inner = int(ssm_ratio * d_model)
        self.dt_rank = int(math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank)
        self.channel_first = channel_first
        self.with_dconv = d_conv > 1
        # self.forward = self.forward

        # todo tags for forward_type =============================="ori"
        # checkpostfix = self.checkpostfix
        # self.disable_force32, forward_type = checkpostfix("_no32", forward_type)
        # self.oact, forward_type = checkpostfix("_oact", forward_type)
        # self.disable_z, forward_type = checkpostfix("_noz", forward_type)
        # self.disable_z_act, forward_type = checkpostfix("_nozact", forward_type)
        # self.out_norm, forward_type = self.get_outnorm(forward_type, self.d_inner, channel_first)
        #todo ====================================================1scan conv concat
        self.disable_force32 = False
        self.oact = False
        self.disable_z = True
        self.disable_z_act = True
        self.out_norm , forward_type= self.get_outnorm(forward_type, self.d_inner, channel_first)
        # forward_type debug =======================================
        FORWARD_TYPES = dict(
            # v01=partial(self.forward_corev2, force_fp32=(not self.disable_force32), selective_scan_backend="mamba",
            #             scan_force_torch=True),
            # v02=partial(self.forward_corev2, force_fp32=(not self.disable_force32), selective_scan_backend="mamba"),
            # v03=partial(self.forward_corev2, force_fp32=(not self.disable_force32), selective_scan_backend="oflex"),
            # v04=partial(self.forward_corev2, force_fp32=False),  # selective_scan_backend="oflex", scan_mode="cross2d"
            # v05=partial(self.forward_corev2, force_fp32=False, no_einsum=True),
            # # selective_scan_backend="oflex", scan_mode="cross2d"
            # # ===============================
            # v051d=partial(self.forward_corev2, force_fp32=False, no_einsum=True, scan_mode="unidi"),
            # v052d=partial(self.forward_corev2, force_fp32=False, no_einsum=True, scan_mode="bidi"),
            # v052dc=partial(self.forward_corev2, force_fp32=False, no_einsum=True, scan_mode="cascade2d"),
            # v052d3=partial(self.forward_corev2, force_fp32=False, no_einsum=True, scan_mode=3),  # debug
            # ===============================
            v2=partial(self.forward_corev2, force_fp32=(not self.disable_force32), selective_scan_backend="core"),
            # v3=partial(self.forward_corev2, force_fp32=False, selective_scan_backend="oflex"),
        )
        self.forward_core = FORWARD_TYPES.get(forward_type, None)

        # in proj =======================================
        d_proj = self.d_inner if self.disable_z else (self.d_inner * 2)
        self.in_proj = Linear(d_model, d_proj, bias=bias, channel_first=channel_first)
        self.act: nn.Module = act_layer()

        # conv =======================================
        if self.with_dconv:
            self.conv2d = nn.Conv2d(
                in_channels=self.d_inner,
                out_channels=self.d_inner,
                groups=self.d_inner,
                bias=conv_bias,
                kernel_size=d_conv,
                padding=(d_conv - 1) // 2,
                **factory_kwargs,
            )

        # x proj ============================
        #todo
        self.x_proj = [nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False)]
        # self.x_proj = Linear(self.d_inner, self.k_group * (self.dt_rank + self.d_state * 2), groups=self.k_group,
        #                      bias=False, channel_first=True)
        self.dt_projs = Linear(self.dt_rank, self.k_group * self.d_inner, groups=self.k_group, bias=False,
                               channel_first=True)

        # self.x_proj = [
        #     nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False)
        #     for _ in range(self.k_group)
        # ]
        # self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0)) # (K, N, inner)
        # del self.x_proj

        # out proj =======================================
        self.out_act = nn.GELU() if self.oact else nn.Identity()
        self.out_proj = Linear(self.d_inner, self.d_model, bias=bias, channel_first=channel_first)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else nn.Identity()

        if initialize in ["v0"]:
            self.A_logs, self.Ds, self.dt_projs_weight, self.dt_projs_bias = mamba_init.init_dt_A_D(
                self.d_state, self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                k_group=self.k_group,
            )
        elif initialize in ["v1"]:
            # simple init dt_projs, A_logs, Ds
            self.Ds = nn.Parameter(torch.ones((self.k_group * self.d_inner)))
            self.A_logs = nn.Parameter(torch.randn(
                (self.k_group * self.d_inner, self.d_state)))  # A == -A_logs.exp() < 0; # 0 < exp(A * dt) < 1
            self.dt_projs_weight = nn.Parameter(
                0.1 * torch.randn((self.k_group, self.d_inner, self.dt_rank)))  # 0.1 is added in 0430
            self.dt_projs_bias = nn.Parameter(0.1 * torch.randn((self.k_group, self.d_inner)))  # 0.1 is added in 0430
        elif initialize in ["v2"]:
            # simple init dt_projs, A_logs, Ds
            self.Ds = nn.Parameter(torch.ones((self.k_group * self.d_inner)))
            self.A_logs = nn.Parameter(torch.zeros(
                (self.k_group * self.d_inner, self.d_state)))  # A == -A_logs.exp() < 0; # 0 < exp(A * dt) < 1

            self.dt_projs_weight = nn.Parameter(0.1 * torch.rand((self.k_group, self.d_inner, self.dt_rank)))
            self.dt_projs_bias = nn.Parameter(0.1 * torch.rand((self.k_group, self.d_inner)))
        self.dt_projs.weight.data = self.dt_projs_weight.data.view(self.dt_projs.weight.shape)
        # self.dt_projs.bias.data = self.dt_projs_bias.data.view(self.dt_projs.bias.shape)
        del self.dt_projs_weight
        # del self.dt_projs_bias
        #todo

        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0))
    def forward_corev2(
            self,
            x: torch.Tensor = None,
            # ==============================
            force_fp32=False,  # True: input fp32
            # ==============================
            ssoflex=True,  # True: input 16 or 32 output 32 False: output dtype as input
            # ==============================
            selective_scan_backend=None,#v2
            # ==============================
            scan_mode="unidi",#cross2d=0, unidi=1, bidi=2, cascade2d=-1
            scan_force_torch=False,
            # ==============================
            **kwargs,
    ):
        # assert selective_scan_backend in [None, "oflex", "mamba", "torch"]
        _scan_mode = dict(cross2d=0, unidi=1, bidi=2, cascade2d=-1).get(scan_mode, None) if isinstance(scan_mode,
                                                                                                       str) else scan_mode  # for debug
        assert isinstance(_scan_mode, int)
        delta_softplus = True
        channel_first = self.channel_first
        to_fp32 = lambda *args: (_a.to(torch.float32) for _a in args)
        force_fp32 = force_fp32 or ((not ssoflex) and self.training)


        B, D, H, W = x[0].shape
        N = self.d_state
        K, D, R = self.k_group, self.d_inner, self.dt_rank
        L = H * W


        def selective_scan(u, delta, A, B, C, D=None, delta_bias=None, delta_softplus=True):
            return selective_scan_fn(u, delta, A, B, C, D, delta_bias, delta_softplus, ssoflex,
                                     backend=selective_scan_backend)

        if True:
            xst = [cross_scan_fn(i, in_channel_first=True, out_channel_first=True, scans=_scan_mode,
                               force_torch=scan_force_torch) for i in x]
            #todo



            xs = x[0].new_empty(B, self.d_inner, L, self.inchannel)
            for j in range(self.inchannel):
                xs[:, :, :, j] = xst[j][:,0]
            xs = xs.flatten(2, 3)


            # xs = x[0].new_empty(B, self.d_inner, L*2)
            # xs[:, :, :L] = xst[0][:,0]
            # xs[:, :, L:2*L] = xst[1][:,0]






            txs = x[0].new_empty((B, 1, self.d_inner, self.inchannel * L))
            txs[:, 0] = xs
            # x_con = self.my_conv(x)
            # # x_con = conv_norm(x_con)
            # x_con = x_con.flatten(2, 3)
            # # x_cons = x.new_empty((B, K,  D+N+N, H * W))#tdo BC
            # x_cons = x.new_empty((B, self.d_inner, H * W, 2))
            # x_cons[:, :, :, 0] = xs[:, 0, :, :]
            # x_cons[:, :, :, 1] = x_con
            # x_conmulty = x_cons.flatten(2, 3)
            # x_consmulty = x.new_empty((B, 1, self.d_inner, 2 * H * W))
            # x_consmulty[:, 0] = x_conmulty

            x_dbl = torch.einsum("b k d l, k c d -> b k c l", txs, self.x_proj_weight)  # 批，方向，矩阵相乘。proj矩阵预测Δ，B，C

            #todo
            # x_dbl = self.x_proj(xs.view(B, -1, L))
            # dts, Bs, Cs = torch.split(x_dbl.contiguous().view(B, K, -1, L), [R, N, N], dim=2)
            dts, Bs, Cs = torch.split(x_dbl, [R, N, N], dim=2)
            dts = dts.contiguous().view(B, -1, L*self.inchannel)
            dts = self.dt_projs(dts)

            xs = xs.view(B, -1, L*self.inchannel)
            dts = dts.contiguous().view(B, -1, L*self.inchannel)
            As = -self.A_logs.to(torch.float).exp()  # (k * c, d_state)
            Ds = self.Ds.to(torch.float)  # (K * c)
            Bs = Bs.contiguous().view(B, K, N, L*self.inchannel)
            Cs = Cs.contiguous().view(B, K, N, L*self.inchannel)
            delta_bias = self.dt_projs_bias.view(-1).to(torch.float)

            if force_fp32:
                xs, dts, Bs, Cs = to_fp32(xs, dts, Bs, Cs)

            yst: torch.Tensor = selective_scan(
                xs, dts, As, Bs, Cs, Ds, delta_bias, delta_softplus
            )
            # todo
            ys = yst.view(B, -1, L, self.inchannel).sum(dim=3)
            # for ti in range(self.inchannel):
            #     ys+=yst[:, :, ti::self.inchannel]
            ys=ys.contiguous().view(B, K, -1, H, W)
            # ys = (ys[:, :, ::self.inchannel] + ys[:, :, 1::self.inchannel]).contiguous().view(B, K, -1, H, W)
            # ys = ys[:, :, ::2].contiguous().view(B, K, -1, H, W)
            y: torch.Tensor = cross_merge_fn(ys, in_channel_first=True, out_channel_first=True, scans=_scan_mode,
                                             force_torch=scan_force_torch)

            if getattr(self, "__DEBUG__", False):
                setattr(self, "__data__", dict(
                    A_logs=self.A_logs, Bs=Bs, Cs=Cs, Ds=Ds,
                    us=xs, dts=dts, delta_bias=delta_bias,
                    ys=ys, y=y, H=H, W=W,
                ))

        y = y.view(B, -1, H, W)
        if not channel_first:
            y = y.permute(0, 2, 3, 1).contiguous()

        y = self.out_norm(y)

        return y.to(x[0].dtype)

    def forward(self, x: torch.Tensor, **kwargs):

        x = [self.in_proj(x[i]) for i in range(self.inchannel)]
        # x = self.in_proj(x)
        # if not self.disable_z:
        #     x, z = x.chunk(2, dim=(1 if self.channel_first else -1))  # (b, h, w, d)
        #     if not self.disable_z_act:
        #         z = self.act(z)
        if not self.channel_first:
            x = x.permute(0, 3, 1, 2).contiguous()
        # if self.with_dconv:
        #     x = self.conv2d(x)  # (b, d, h, w)
        # x = self.act(x)
        x = [self.act(x[i]) for i in range(self.inchannel)]
        y = self.forward_core(x)
        y = self.out_act(y)
        # if not self.disable_z:
        #     y = y * z
        out = self.dropout(self.out_proj(y))
        return out

    @staticmethod
    def get_outnorm(forward_type="", d_inner=192, channel_first=True):
        def checkpostfix(tag, value):
            ret = value[-len(tag):] == tag
            if ret:
                value = value[:-len(tag)]
            return ret, value

        out_norm_none, forward_type = checkpostfix("_onnone", forward_type)
        out_norm_dwconv3, forward_type = checkpostfix("_ondwconv3", forward_type)
        out_norm_cnorm, forward_type = checkpostfix("_oncnorm", forward_type)
        out_norm_softmax, forward_type = checkpostfix("_onsoftmax", forward_type)
        out_norm_sigmoid, forward_type = checkpostfix("_onsigmoid", forward_type)

        out_norm = nn.Identity()
        if out_norm_none:
            out_norm = nn.Identity()
        elif out_norm_cnorm:
            out_norm = nn.Sequential(
                LayerNorm(d_inner, channel_first=channel_first),
                (nn.Identity() if channel_first else Permute(0, 3, 1, 2)),
                nn.Conv2d(d_inner, d_inner, kernel_size=3, padding=1, groups=d_inner, bias=False),
                (nn.Identity() if channel_first else Permute(0, 2, 3, 1)),
            )
        elif out_norm_dwconv3:
            out_norm = nn.Sequential(
                (nn.Identity() if channel_first else Permute(0, 3, 1, 2)),
                nn.Conv2d(d_inner, d_inner, kernel_size=3, padding=1, groups=d_inner, bias=False),
                (nn.Identity() if channel_first else Permute(0, 2, 3, 1)),
            )
        elif out_norm_softmax:
            out_norm = SoftmaxSpatial(dim=(-1 if channel_first else 1))
        elif out_norm_sigmoid:
            out_norm = nn.Sigmoid()
        else:
            out_norm = LayerNorm(d_inner, channel_first=channel_first)

        return out_norm, forward_type

    @staticmethod
    def checkpostfix(tag, value):
        ret = value[-len(tag):] == tag
        if ret:
            value = value[:-len(tag)]
        return ret, value
class SpatioTemporalGuidance(nn.Module):
    """
    时空引导模块：将 Stem 层提取的 N 帧混合特征，
    通过自适应下采样注入到深层网络中。
    """

    def __init__(self, in_channels, out_channels, scale_factor):
        super().__init__()
        # 1. 空间对齐：根据目标层级（P3/P4/P5）进行下采样
        if scale_factor > 1:
            self.downsample = nn.Sequential(
                *[Conv(in_channels, in_channels, k=3, s=2) for _ in range(int(torch.log2(torch.tensor(scale_factor))))]
            )
        else:
            self.downsample = nn.Identity()

        # 2. 特征映射：调整通道数以匹配目标层
        self.map = Conv(in_channels, out_channels, k=1)

    def forward(self, stem_feat, target_feat):
        # stem_feat: 来自强化 Stem 层的输出
        # target_feat: 来自当前层级 (如 P3) 的特征层
        guide = self.downsample(stem_feat)
        guide = self.map(guide)

        # 使用 Concat 进行引导注入
        return torch.cat([target_feat, guide], dim=1)



def autopad(k, p=None, d=1):
    """
    自动计算填充 (Padding) 尺寸，以保持输入输出的特征图分辨率一致 (Same Padding)。

    参数:
        k (int | tuple): 卷积核大小 (Kernel size)。
        p (int | tuple | None): 用户自定义的 padding 大小。如果为 None，则自动计算。
        d (int | tuple): 膨胀率 (Dilation)。
    返回:
        自动计算出的 padding 大小。
    """
    # 针对 3D 卷积，k 可以是整数或包含三个元素的元组
    if d > 1:
        # 考虑膨胀卷积的情况，重新计算实际的卷积核感受野大小
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]
    if p is None:
        # 根据实际卷积核大小计算 padding (假设 stride=1 时保持尺寸不变)
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p


class Conv3dBlock(nn.Module):
    """
    标准 3D 卷积模块 (Standard 3D Convolution Block)。
    包含: Conv3d + BatchNorm3d + SiLU (默认激活函数)。
    处理张量维度: (B, C, D, H, W)
    """

    def __init__(self, c1, c2, k=3, s=1, p=None, g=1, d=1, act=True):
        """
        初始化 3D 卷积块。

        参数:
            c1 (int): 输入通道数 (Input channels)。
            c2 (int): 输出通道数 (Output channels)。
            k (int | tuple): 卷积核大小 (Kernel size)，默认为 3。
            s (int | tuple): 步长 (Stride)，默认为 1。
            p (int | tuple | None): 填充 (Padding)，默认通过 autopad 自动计算。
            g (int): 分组卷积的组数 (Groups)，默认为 1。
            d (int | tuple): 膨胀率 (Dilation)，默认为 1。
            act (bool | nn.Module): 是否使用激活函数，或传入自定义的激活函数实例。
        """
        super().__init__()

        # 核心 3D 卷积层
        # 注意: 当后接 BatchNorm 时，卷积层的 bias 必须设为 False 以节省显存并避免冗余计算
        self.conv = nn.Conv3d(
            in_channels=c1,
            out_channels=c2,
            kernel_size=k,
            stride=s,
            padding=autopad(k, p, d),
            groups=g,
            dilation=d,
            bias=False
        )

        # 3D 批量归一化层
        self.bn = nn.BatchNorm2d(num_features=c2)

        # 激活函数配置
        # 默认使用 SiLU (Swish)，这是 YOLO 架构中常用的激活函数，在深度网络中表现优异
        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        """
        前向传播 (Forward pass)。

        参数:
            x (torch.Tensor): 输入张量，形状必须为 5D: (Batch_size, Channels, Depth, Height, Width)。
        返回:
            torch.Tensor: 输出张量。
        """
        x = x.unsqueeze(1)
        x = self.conv(x)
        x = self.pool_to_4d(x)
        x = self.bn(x)
        x = self.act(x)

        return x
    def pool_to_4d(self,x, mode='mean'):
        """
        沿着深度维度进行池化压缩
        """
        # 假设输入形状为 [1, 64, 5, 128, 128]
        if mode == 'mean':
            # 沿索引为 2 的维度 (Depth) 求均值
            x_pooled = torch.mean(x, dim=2)
        elif mode == 'max':
            # 沿索引为 2 的维度求最大值，torch.max 会返回 (values, indices)
            x_pooled, _ = torch.max(x, dim=2)
        else:
            raise ValueError("不支持的池化模式")

        # 输出形状保持为 (B, C, H, W) -> [1, 64, 128, 128]
        return x_pooled
