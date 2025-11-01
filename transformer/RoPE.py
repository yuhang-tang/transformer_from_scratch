import torch
# 旋转位置编码
def build_rope_frequencies(dim, base=10000):
    half_dim = dim // 2
    return 1.0 / (base ** (torch.arange(half_dim, dtype=torch.float32) * 2 / dim))

def build_rope_theta(seqlen, dim, base=10000):
    position = torch.arange(seqlen,dtype=torch.float32)
    freq = build_rope_frequencies(dim, base)
    return torch.outer(position, freq) # torch.outer(a, b) 会计算所有 a_i * b_j 的组合，结果为一个矩阵

def build_cos_sin(seqlen, dim, base=10000):
    theta = build_rope_theta(seqlen, dim, base)
    return torch.cos(theta), torch.sin(theta) # 维度是[seqlen, dim//2]

def apply_rotary_position(x, cos, sin):
    x1 = x[...,::2]
    x2 = x[...,1::2]
    # 让x和cos，sin维度相同；因为cos的维度固定为[seqlen, dim // 2],则x和cos这两个维度进行对齐，增加x比cos多的其他维度为1即可进行广播
    shape = [d if i == 1 or i == x.ndim - 1 else 1 for i,d in enumerate(x.shape)]
    cos = cos.view(shape)
    sin = sin.view(shape)
    x_rotated = torch.cat([
        x1 * cos - x2 * sin, # 这里需要x1与cos，sin的维度对齐才能够相乘 
        x1 * sin + x2 * cos
    ], dim=-1)
    return x_rotated

class RoPE:
    def __init__(self, x, seqlen, dim, base=10000):
        self.seqlen = seqlen
        self.dim = dim
        self.base = base
        self.x = x
    def build_cos_sin(self):
        half_dim = self.dim // 2
        freq = torch.ones(1, device=self.x.device) / (self.base ** (torch.arange(half_dim, dtype=torch.float32, device=self.x.device) * 2 / self.dim))
        position = torch.arange(self.seqlen, dtype=torch.float32, device=self.x.device)
        theta = torch.outer(position, freq)
        cos, sin = torch.cos(theta), torch.sin(theta)
        # x和cos的最后两个维度要对齐
        # 这里假设了x的输入维度为[..., seqlen, dim]
        while cos.ndim < self.x.ndim:
            cos = cos.unsqueeze(0)
            sin = sin.unsqueeze(0)
        return cos, sin
    def forward(self):
        cos, sin = self.build_cos_sin()
        x1 = self.x[..., ::2]
        x2 = self.x[...,1::2]
        return torch.cat([
            x1 * cos - x2 * sin,
            x1 * sin + x2 * cos
        ], dim=-1)
x = torch.randn(2, 10, 128)
rope = RoPE(x, 10, 128)
output = rope.forward()
print(f"输入形状: {x.shape}")
print(f"输出形状: {output.shape}")
print(f"预期输出形状: torch.Size([2, 10, 128])")
print(f"验证通过: {output.shape == x.shape}")

# 多头的验证：
x_multi = torch.randn(2, 8, 10, 16)  # 假设8个头，每个头16维； x要和cos的最后两个维度对齐[seqlen, dim // 2]
rope_multi = RoPE(x_multi, 10, 16)
output_multi = rope_multi.forward()
print(f"多头输入形状: {x_multi.shape}")
print(f"多头输出形状: {output_multi.shape}")
print(f"预期输出形状: torch.Size([2, 10, 8, 128])")
print(f"多头验证通过: {output_multi.shape == x_multi.shape}")