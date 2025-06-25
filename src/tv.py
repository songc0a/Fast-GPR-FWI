import torch

def denoising_2D_TV(data: torch.Tensor) -> torch.Tensor:
    data_max = torch.max(data)
    data=data.squeeze(-1)
    data = data / (data_max + 1e-6)
    M, N = data.shape
    X0 = torch.zeros((M + 2, N + 2), device=data.device)
    X0[1:M + 1, 1:N + 1] = data
    Y0 = X0.clone()
    X = torch.zeros_like(X0)
    Zx = torch.zeros_like(X0)
    Zy = torch.zeros_like(X0)
    Ux = torch.zeros_like(X0)
    Uy = torch.zeros_like(X0)

    lamda = 0.02
    rho_ = 1.0
    num = 500
    err = 1e-5

    return_data = denoising_2D_TV_(num, X, X0, err, M, N, Zx, Zy, Ux, Uy, Y0, lamda, rho_)
    return return_data * data_max


def denoising_2D_TV_(num, X, X0, err, M, N, Zx, Zy, Ux, Uy, Y0, lamda, rho_):
    K = 0
    while K < num and torch.norm(X - X0, p=2) > err:
        X0 = X.clone()
        MM, NN = M + 2, N + 2

        Dxt_Zx = torch.zeros((MM, NN), device=X.device)
        Dyt_Zy = torch.zeros_like(Dxt_Zx)
        Dxt_Ux = torch.zeros_like(Dxt_Zx)
        Dyt_Uy = torch.zeros_like(Dxt_Zx)

        # Compute gradients
        Dxt_Zx[:, :-1] = Zx[:, :-1] - Zx[:, 1:]
        Dxt_Zx[:, -1] = Zx[:, -1] - Zx[:, 0]

        Dyt_Zy[:-1, :] = Zy[:-1, :] - Zy[1:, :]
        Dyt_Zy[-1, :] = Zy[-1, :] - Zy[0, :]

        Dxt_Ux[:, :-1] = Ux[:, :-1] - Ux[:, 1:]
        Dxt_Ux[:, -1] = Ux[:, -1] - Ux[:, 0]

        Dyt_Uy[:-1, :] = Uy[:-1, :] - Uy[1:, :]
        Dyt_Uy[-1, :] = Uy[-1, :] - Uy[0, :]

        RHS = Y0 + lamda * rho_ * (Dxt_Zx + Dyt_Zy) - lamda * (Dxt_Ux + Dyt_Uy)

        X = torch.zeros_like(X0)
        X[1:M + 1, 1:N + 1] = ((X0[2:M + 2, 1:N + 1] + X0[0:M, 1:N + 1] + X0[1:M + 1, 2:N + 2] + X0[1:M + 1,
                                                                                                 0:N]) * lamda * rho_ + RHS[
                                                                                                                        1:M + 1,
                                                                                                                        1:N + 1]) / (
                                          1 + 4 * lamda * rho_)

        # Compute new Z
        Dx_X = torch.zeros_like(X)
        Dy_X = torch.zeros_like(X)

        Dx_X[:, 1:] = X[:, 1:] - X[:, :-1]
        Dx_X[:, 0] = X[:, 0] - X[:, -1]

        Dy_X[1:, :] = X[1:, :] - X[:-1, :]
        Dy_X[0, :] = X[0, :] - X[-1, :]

        Tx = Ux / rho_ + Dx_X
        Ty = Uy / rho_ + Dy_X

        Zx = torch.sign(Tx) * torch.clamp(torch.abs(Tx) - 1 / rho_, min=0)
        Zy = torch.sign(Ty) * torch.clamp(torch.abs(Ty) - 1 / rho_, min=0)

        # Update U
        Ux = Ux + (Dx_X - Zx)
        Uy = Uy + (Dy_X - Zy)

        K += 1

    return X[1:M + 1, 1:N + 1]

import torch.nn.functional as F
def total_variation1(epsilon, beta=1e-6, anisotropic=False):
    """
    计算TV正则项
    :param epsilon: 反演变量 (tensor)
    :param beta: 避免梯度爆炸的小常数
    :param anisotropic: 是否使用各向异性 TV
    :return: TV 正则项
    """
    epsilon.squeeze_(-1)
    dx1 = epsilon[:-1, :] - epsilon[1:, :]  # x 方向梯度
    dy1 = epsilon[:, :-1] - epsilon[:, 1:]  # y 方向梯度
    dx=F.pad(dx1, (0, 0, 0, 1), mode="constant", value=0)
    dy=F.pad(dy1, (0, 1, 0, 0), mode="constant", value=0)
    print(dx.shape)
    print(dy.shape)
    if anisotropic:
        tv = torch.abs(dx).sum() + torch.abs(dy).sum()
    else:
        tv = torch.sqrt(dx ** 2 + dy ** 2 + beta).sum()
    print(tv)
    print(tv.shape)
    tv.unsqueeze_(-1)

    return tv


def total_variation2(epsilon, beta=1e-6):
    epsilon.squeeze_(-1)
    x, y = epsilon.shape
    grad_tv = torch.zeros_like(epsilon)
    # 计算 TV 梯度 (对右边和下边计算)
    dx = epsilon[:-1, :] - epsilon[1:, :]
    dy = epsilon[:, :-1] - epsilon[:, 1:]
    # 计算梯度分量
    dx_grad = dx / torch.sqrt(dx**2 + beta)  # d(TV)/d(epsilon_{i,j}) 来自 (i+1, j)
    dy_grad = dy / torch.sqrt(dy**2 + beta)  # d(TV)/d(epsilon_{i,j}) 来自 (i, j+1)

    # 加入梯度贡献
    grad_tv[:-1, :] += dx_grad
    grad_tv[1:, :] -= dx_grad  # 反向贡献
    grad_tv[:, :-1] += dy_grad
    grad_tv[:, 1:] -= dy_grad  # 反向贡献
    grad_tv.unsqueeze_(-1)

    return grad_tv


import torch
import torch.nn.functional as F


def total_variation(data, lamda=0.02, rho=1.0, num_iter=500, tol=1e-5):
    # 数据归一化

    data.squeeze_(-1)
    data_max = data.max()
    normalized_data = data / (data_max + 1e-6)
    M, N = normalized_data.shape

    # 初始化变量（带边界）
    X = torch.zeros((M + 2, N + 2), device=data.device)
    X[1:-1, 1:-1] = normalized_data
    Y = X.clone()

    # 辅助变量和乘子
    Zx = torch.zeros_like(X)
    Zy = torch.zeros_like(X)
    Ux = torch.zeros_like(X)
    Uy = torch.zeros_like(X)

    # 创建四邻域卷积核
    kernel = torch.tensor([[0, 1, 0],
                           [1, 0, 1],
                           [0, 1, 0]], dtype=torch.float32, device=data.device).view(1, 1, 3, 3)

    for _ in range(num_iter):
        X_prev = X.clone()

        # 计算差分项
        # 水平方向差分 (循环边界)
        Dxt_Zx = torch.zeros_like(Zx)
        Dxt_Zx[:, :-1] = Zx[:, :-1] - Zx[:, 1:]
        Dxt_Zx[:, -1] = Zx[:, -1] - Zx[:, 0]

        # 垂直方向差分 (循环边界)
        Dyt_Zy = torch.zeros_like(Zy)
        Dyt_Zy[:-1, :] = Zy[:-1, :] - Zy[1:, :]
        Dyt_Zy[-1, :] = Zy[-1, :] - Zy[0, :]

        # 乘子项的差分
        Dxt_Ux = torch.zeros_like(Ux)
        Dxt_Ux[:, :-1] = Ux[:, :-1] - Ux[:, 1:]
        Dxt_Ux[:, -1] = Ux[:, -1] - Ux[:, 0]

        Dyt_Uy = torch.zeros_like(Uy)
        Dyt_Uy[:-1, :] = Uy[:-1, :] - Uy[1:, :]
        Dyt_Uy[-1, :] = Uy[-1, :] - Uy[0, :]

        # 构建RHS
        RHS = Y + lamda * rho * (Dxt_Zx + Dyt_Zy) - lamda * (Dxt_Ux + Dyt_Uy)

        # 使用卷积进行邻域平均
        neighbor_sum = F.conv2d(X[None, None, :, :], kernel, padding=0).squeeze()
        X_center = (neighbor_sum * lamda * rho + RHS[1:-1, 1:-1]) / (1 + 4 * lamda * rho)

        # 更新X的中间区域
        X[1:-1, 1:-1] = X_center

        # 计算梯度项 (带循环边界)
        Dx_X = torch.zeros_like(X)
        Dx_X[:, 1:] = X[:, 1:] - X[:, :-1]
        Dx_X[:, 0] = X[:, 0] - X[:, -1]

        Dy_X = torch.zeros_like(X)
        Dy_X[1:, :] = X[1:, :] - X[:-1, :]
        Dy_X[0, :] = X[0, :] - X[-1, :]

        # 更新Z变量 (软阈值)
        Tx = (Ux + rho * Dx_X) / rho
        Zx = torch.sign(Tx) * torch.clamp(torch.abs(Tx) - 1 / rho, min=0)

        Ty = (Uy + rho * Dy_X) / rho
        Zy = torch.sign(Ty) * torch.clamp(torch.abs(Ty) - 1 / rho, min=0)

        # 更新乘子
        Ux += rho * (Dx_X - Zx)
        Uy += rho * (Dy_X - Zy)

        # 收敛判断
        if torch.norm(X - X_prev) < tol:
            break

    # 返回去噪结果并恢复原始范围
    return (X[1:-1, 1:-1] * data_max).unsqueeze(-1)