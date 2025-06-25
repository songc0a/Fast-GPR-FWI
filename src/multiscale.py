
import torch

def design_fir_filter(cutoff: float, fs: float, numtaps: int) -> torch.Tensor:
    """使用 Hamming 窗设计 FIR 滤波器"""
    n = torch.arange(numtaps, dtype=torch.float32)
    # 汉明窗
    window = 0.54 - 0.46 * torch.cos(2 * torch.pi * n / (numtaps - 1))
    # 正弦波
    sinc = torch.sin(2 * torch.pi * (cutoff/fs) * (n - (numtaps-1)/2)) / (torch.pi * (n - (numtaps-1)/2))
    # 处理中心点
    center = (numtaps-1) // 2
    sinc[center] = 2 * cutoff/fs
    # 应用窗函数
    h = window * sinc
    # 归一化
    return h / h.sum()

def apply_filter(data: torch.Tensor, fs: float, cutoff: float) -> torch.Tensor:
    """应用 FIR 滤波器到数据"""
    numtaps = int(1 * (fs / cutoff))
    fir_coeff = design_fir_filter(cutoff, fs, numtaps)
    fir_coeff = fir_coeff.to(data.device)

    if data.ndim == 1:
        # 1D 数据处理 - 添加维度以支持反射填充
        data_2d = data.view(1, 1, -1)
        padded_data = torch.nn.functional.pad(data_2d, (numtaps-1, 0), mode='reflect')
        filtered = torch.nn.functional.conv1d(
            padded_data, 
            fir_coeff.view(1, 1, -1), 
            padding=0
        )
        return filtered.view(-1)

    elif data.ndim == 3:
        # 3D 数据处理
        step, iterations, nrx = data.shape
        # 重塑数据以使用批量处理
        reshaped_data = data.permute(0, 2, 1).reshape(-1, 1, iterations)
        # 填充数据
        padded_data = torch.nn.functional.pad(reshaped_data, (numtaps-1, 0), mode='reflect')
        # 应用卷积
        filtered = torch.nn.functional.conv1d(
            padded_data,
            fir_coeff.view(1, 1, -1),
            padding=0
        )
        # 重塑回原始形状
        return filtered.view(step, nrx, iterations).permute(0, 2, 1)

    else:
        raise ValueError(f"不支持的数据维度: {data.ndim}。期望 1D 或 3D tensor。")