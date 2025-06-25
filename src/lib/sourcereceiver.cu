#include <cuda_runtime.h>
#include <iostream>
#include <stdio.h>

#define INDEX4D_FIELDS(s, i, j, k, NX, NY, NZ) ((s)*(NX)*(NY)*(NZ) + (i)*(NY)*(NZ) + (j)*(NZ) + (k))
#define INDEX4D_RXS(s, c, t, rx, NY_RXS, N_ITER, NRX) ((s)*(NY_RXS)*(N_ITER)*(NRX) + (c)*(N_ITER)*(NRX) + (t)*(NRX) + (rx))
#define INDEX3D_RXCOORDS(s, rx, d, NRX, DIM) ((s)*(NRX)*(DIM) + (rx)*(DIM) + (d))

extern "C" {

__global__ void store_outputs(
    int step, int NRX, int iteration, const int* __restrict__ rxcoords,
    float *rxs, const float* __restrict__ Ex, const float* __restrict__ Ey,
    const float* __restrict__ Ez, const float* __restrict__ Hx,
    const float* __restrict__ Hy, const float* __restrict__ Hz,
    int NX, int NY, int NZ, int DIM, int NY_RXS, int N_ITER
) {
    int rx = blockIdx.x * blockDim.x + threadIdx.x;
    int s = blockIdx.y * blockDim.y + threadIdx.y;

    if (rx < NRX && s < step) {

        // 提取 rxcoords 的坐标
        int i = rxcoords[INDEX3D_RXCOORDS(s, rx, 0, NRX, DIM)];
        int j = rxcoords[INDEX3D_RXCOORDS(s, rx, 1, NRX, DIM)];
        int k = rxcoords[INDEX3D_RXCOORDS(s, rx, 2, NRX, DIM)];
        // 存储电场分量
        rxs[INDEX4D_RXS(s, 0, iteration, rx, NY_RXS, N_ITER, NRX)] = Ex[INDEX4D_FIELDS(s, i, j, k, NX, NY, NZ)];
        rxs[INDEX4D_RXS(s, 1, iteration, rx, NY_RXS, N_ITER, NRX)] = Ey[INDEX4D_FIELDS(s, i, j, k, NX, NY, NZ)];
        rxs[INDEX4D_RXS(s, 2, iteration, rx, NY_RXS, N_ITER, NRX)] = Ez[INDEX4D_FIELDS(s, i, j, k, NX, NY, NZ)];
        // 存储磁场分量
        rxs[INDEX4D_RXS(s, 3, iteration, rx, NY_RXS, N_ITER, NRX)] = Hx[INDEX4D_FIELDS(s, i, j, k, NX, NY, NZ)];
        rxs[INDEX4D_RXS(s, 4, iteration, rx, NY_RXS, N_ITER, NRX)] = Hy[INDEX4D_FIELDS(s, i, j, k, NX, NY, NZ)];
        rxs[INDEX4D_RXS(s, 5, iteration, rx, NY_RXS, N_ITER, NRX)] = Hz[INDEX4D_FIELDS(s, i, j, k, NX, NY, NZ)];
    }
}


// 优化后的接口函数定义
void launch_store_outputs(
    int step, int NRX, int iteration, const int* __restrict__ rxcoords,
    float *rxs, const float* __restrict__ Ex, const float* __restrict__ Ey,
    const float* __restrict__ Ez, const float* __restrict__ Hx,
    const float* __restrict__ Hy, const float* __restrict__ Hz,
    int NX, int NY, int NZ, int DIM, int NY_RXS, int N_ITER//,float *grader,float *gradse,float *graderrx,float *gradserx
) {
    dim3 threads_per_block(32, 4);  // 每个线程块负责 32 个接收器和 4 个步长
    dim3 num_blocks((NRX + 31) / 32, (step + 3) / 4);  // 网格大小

    store_outputs<<<num_blocks, threads_per_block>>>(
        step, NRX, iteration, rxcoords, rxs, Ex, Ey, Ez, Hx, Hy, Hz,
        NX, NY, NZ, DIM, NY_RXS, N_ITER//,grader,gradse,graderrx,gradserx
    );
    // 使用 cudaDeviceSynchronize() 确保内核执行完成
    cudaDeviceSynchronize();
}


__global__ void Update_hertzian_dipole(
    int step, int iteration, float dx, float dy, float dz,
    const int* __restrict__ srcinfo1, const float srcinfo2, const float* __restrict__ srcwaveforms,
    float* Ex, float* Ey, float* Ez, float* uE4,
    int NX, int NY, int NZ, int NY_SRCINFO, int polarisation,int iterations
) {
    int src = blockIdx.x * blockDim.x + threadIdx.x; // 对应源维度
    int s = blockIdx.y * blockDim.y + threadIdx.y;   // 对应 step 维度

    if (src < NY_SRCINFO && s < step) {
        // 从 srcinfo1 中获取源位置信息 (i, j, k)
        int i = srcinfo1[s * NY_SRCINFO * 3 + src * 3 + 0];
        int j = srcinfo1[s * NY_SRCINFO * 3 + src * 3 + 1];
        int k = srcinfo1[s * NY_SRCINFO * 3 + src * 3 + 2];
        float dl = srcinfo2; // 每个源可能有不同的 dl 值
        float waveform_value = srcwaveforms[src * iterations + iteration];// 获取第 src 个源在当前迭代的波形值
        float scale = waveform_value * dl / (dx * dy * dz);

            if (polarisation == 0) {
                Ex[INDEX4D_FIELDS(s, i, j, k, NX, NY, NZ)] -= uE4[INDEX4D_FIELDS(s, i, j, k, NX, NY, NZ)] * scale;
            }
            else if (polarisation == 1) {
                Ey[INDEX4D_FIELDS(s, i, j, k, NX, NY, NZ)] -= uE4[INDEX4D_FIELDS(s, i, j, k, NX, NY, NZ)] * scale;
            }
            else if (polarisation == 2){
                Ez[INDEX4D_FIELDS(s, i, j, k, NX, NY, NZ)] -= uE4[INDEX4D_FIELDS(s, i, j, k, NX, NY, NZ)] * scale;
            }
        }
    }

void update_hertzian_dipole(
    int step, int iteration, float dx, float dy, float dz,
    const int* __restrict__ srcinfo1, const float srcinfo2, const float* __restrict__ srcwaveforms,
    float* Ex, float* Ey, float* Ez, float* uE4,
    int NX, int NY, int NZ, int NY_SRCINFO, int polarisation,int iterations
) {
    dim3 threadsPerBlock(16, 16); // 每个块 16x16 线程
    dim3 numBlocks((NY_SRCINFO + 15) / 16, (step + 15) / 16); // 按源和 step 分块

    Update_hertzian_dipole<<<numBlocks, threadsPerBlock>>>(
        step, iteration, dx, dy, dz,
        srcinfo1, srcinfo2, srcwaveforms,
        Ex, Ey, Ez, uE4, NX, NY, NZ, NY_SRCINFO, polarisation,iterations
    );
    cudaDeviceSynchronize();
}


}

