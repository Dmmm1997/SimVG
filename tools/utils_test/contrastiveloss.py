import torch
import torch.nn.functional as F

def contrastive_loss(A, B):
    N, C = A.size()

    # 计算每对特征之间的欧氏距离的平方
    dists = torch.norm(A - B, dim=1, p=2)**2

    # 构建标签（1表示正样本，0表示负样本）
    labels = torch.eye(N).to(A.device)

    # 计算损失
    loss = 0.5 * (1 - labels) * dists + 0.5 * labels * F.relu(1 - dists)

    return loss.mean()

# 示例用法
N, C = 5, 3  # 举例：N=5, C=3
A = torch.randn(N, C)  # 随机生成A张量
B = torch.randn(N, C)  # 随机生成B张量

# 计算对比学习损失
loss = contrastive_loss(A, B)
print(loss)
