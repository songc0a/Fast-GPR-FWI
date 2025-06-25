import numpy as np
import matplotlib.pyplot as plt
import torch
import os
from torch import Tensor
import torch.nn.functional as F

def savebscan(observed_data,n,save_dir = 'visualization',filename="Forward"):
    plt.figure(figsize=(4, 6))
    data_slice = np.rot90(torch.flip(observed_data[:, :, n], dims=[0]).cpu().detach().numpy(), k=-1)
    max_abs = np.max(np.abs(data_slice))
    ny1, nx1 = data_slice.shape
    im = plt.imshow(data_slice,
                    cmap='gray',  # 使用seismic色图以便在0附近使用对比色
                    aspect='auto',
                    # extent=[0, nx1, ny1 * tt, 0],  # y轴乘以tt并翻转方向
                    vmin=-50,  # 对称的色标范围
                    vmax=50,
                    origin='upper')  # 设置原点在上方

    plt.colorbar(im, label='Field strength (V/m)')
    plt.title('Ez')
    plt.xlabel('Trace number')
    plt.ylabel('Time (s)')
    plt.gca().yaxis.set_major_formatter(plt.ScalarFormatter(useMathText=True))
    plt.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, filename))


def save_parameter_plot(data, title, label, filename,save_dir = 'visualization',cut=0):
    if cut!=0:
        data=data[cut:-cut,cut:-cut,:]
    plt.figure(figsize=(5, 6), dpi=300)
    plt.imshow(np.rot90(data.squeeze(-1).detach().cpu().numpy()), aspect='auto')
    plt.title(title)
    plt.colorbar(label=label)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, filename))
    plt.close()

def saveloss(n_epochs,losses,save_dir = 'visualization'):
    # 绘制loss曲线
    plt.figure(figsize=(10, 6), dpi=300)
    plt.plot(range(n_epochs), losses, 'b-', linewidth=2)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss over Epochs')
    plt.gca().yaxis.set_major_formatter(plt.ScalarFormatter(useMathText=True))
    plt.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
    plt.savefig(os.path.join(save_dir, 'loss_curve.png'))
    plt.close()

def visualize_field(G, iteration, save_dir='field_visualizations/forward', reverse=False):
    """保存所有时刻的Ez波场为GIF动画"""
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    from PIL import Image
    
    # 创建保存目录
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # 用于存储所有图像
    images = []
    
    # 获取场的最大值以固定colorbar范围
    max_val = abs(G.Ez_gpu).max().item()
    vmin, vmax = -max_val, max_val
    
    # 为每个时间步创建图像
    for t in range(G.iterations):
        # 获取当前时间步的Ez场
        matrix_2d = G.Ez_gpu[10, :G.nx, :G.ny, 0].cpu().numpy()
        matrix_2d = np.rot90(matrix_2d, k=1)
        
        # 创建图像
        plt.figure(figsize=(8, 6))
        plt.imshow(matrix_2d, 
                  cmap='seismic',
                  aspect='auto',
                  vmin=vmin, 
                  vmax=vmax)
        plt.colorbar(label='Ez field (V/m)')
        plt.title(f'Ez field at t={t*G.dt:.2e}s')
        plt.xlabel('x (m)')
        plt.ylabel('y (m)')
        
        # 保存当前帧
        frame_path = os.path.join(save_dir, f'frame_{t:04d}.png')
        plt.savefig(frame_path)
        plt.close()
        
        # 读取保存的图像并添加到列表
        images.append(Image.open(frame_path))
    
    # 保存GIF动画
    images[0].save(
        os.path.join(save_dir, 'Ez_field.gif'),
        save_all=True,
        append_images=images[1:],
        duration=50,  # 每帧持续时间(ms)
        loop=0       # 0表示无限循环
    )
    
    # 清理临时PNG文件
    for t in range(G.iterations):
        os.remove(os.path.join(save_dir, f'frame_{t:04d}.png'))
    
    print(f"Animation saved to {os.path.join(save_dir, 'Ez_field.gif')}")

def calculatessim(tensor3, tensor4, window_size=11, C1=0.01 ** 2, C2=0.03 ** 2):
    """
    计算两个 2D Tensor 之间的 SSIM (结构相似性指数)
    """
    tensor1=tensor3.squeeze(-1).clone().to(dtype=torch.float32)
    tensor2=tensor4.squeeze(-1).clone().to(dtype=torch.float32)
    assert tensor1.shape == tensor2.shape, "输入张量的形状必须相同"

    # 计算均值
    pad = window_size // 2
    kernel = torch.ones((1, 1, window_size, window_size), device=tensor1.device) / (window_size ** 2)

    mu1 = F.conv2d(tensor1.unsqueeze(0).unsqueeze(0), kernel, padding=pad)
    mu2 = F.conv2d(tensor2.unsqueeze(0).unsqueeze(0), kernel, padding=pad)

    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2

    # 计算方差和协方差
    sigma1_sq = F.conv2d(tensor1.unsqueeze(0).unsqueeze(0) ** 2, kernel, padding=pad) - mu1_sq
    sigma2_sq = F.conv2d(tensor2.unsqueeze(0).unsqueeze(0) ** 2, kernel, padding=pad) - mu2_sq
    sigma12 = F.conv2d((tensor1 * tensor2).unsqueeze(0).unsqueeze(0), kernel, padding=pad) - mu1_mu2

    # 计算 SSIM
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    return ssim_map.mean().item()


def plot_er_with_source(er, source_tensor, save_path='er_with_source.png'):

    er_np = er.detach().cpu().numpy()
    if er_np.ndim == 3:
        er_np = er_np[:, :, 0]
    plt.figure(figsize=(10, 4))
    plt.imshow(er_np, cmap='jet', aspect='auto')
    plt.colorbar(label='epsilon')
    if source_tensor.ndim == 3:
        source_np = source_tensor[:, 0, :].cpu().numpy()
    else:
        source_np = source_tensor[:, :2].cpu().numpy()
    plt.scatter(source_np[:, 1], source_np[:, 0], marker='*', c='red', s=60, label='Source')
    plt.legend()
    plt.title('er with source positions')
    plt.xlabel('Y')
    plt.ylabel('X')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
