
import math
class CosineAnnealingRestartLR:
    def __init__(self, lr_min, lr_max, iter, total_iter, lr_decay_iter, warmup_iters):
        self.lr_min = lr_min
        self.lr_max = lr_max
        self.total_iter = total_iter
        self.lr_decay_iter = lr_decay_iter
        self.warmup_iters = warmup_iters
        self.current_iter = iter # 当前的轮次可以由外部传入；也可以通过step递增
    
    def get_lr(self):
        if self.current_iter < self.warmup_iters:
            return self.lr_max * self.current_iter / self.warmup_iters
        # 超出退火阶段
        if self.current_iter > self.lr_decay_iter:
            return self.lr_min
        
        # 退火阶段
        decay_ratio = (self.current_iter - self.warmup_iters) / (self.total_iter - self.warmup_iters)  # 有warmup阶段，需要从warmup阶段结束后作为退火起点
        cosine = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return self.lr_min + cosine * (self.lr_max - self.lr_min)
    
    def step(self):
        self.current_iter += 1
        return self.get_lr()

# 使用余弦退火，格局迭代次数选择学习率
def get_lr(self, lr_min, lr_max, iter, total_iter, lr_decay_iter, warmup_iters):
    # warmup阶段
    if iter < warmup_iters:
        return lr_max * iter / warmup_iters
    # 超出退火阶段
    if iter > lr_decay_iter:
        return lr_min
    
    # 退火阶段
    decay_ratio = (iter - warmup_iters) / (total_iter - warmup_iters)  # 有warmup阶段，需要从warmup阶段结束后作为退火起点
    cosine = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return lr_min + cosine * (lr_max - lr_min)

lr_min = 0.01 
lr_max = 0.1
iter = 60
total_iter = 100
lr_decay_iter = 90
warmup_iters = 10

sample = CosineAnnealingRestartLR(lr_min, lr_max=lr_max, iter=iter, total_iter=total_iter, lr_decay_iter=lr_decay_iter, warmup_iters=warmup_iters)
for _ in range(total_iter):
    lr = sample.step()
    print(lr)
