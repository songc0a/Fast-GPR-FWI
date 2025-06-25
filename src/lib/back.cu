#include <cuda_runtime.h>
#include <iostream>
#include <stdio.h>
#define INDEX4D(t, i, j, k, NX, NY, NZ) ((t)*(NX)*(NY)*(NZ) + (i)*(NY)*(NZ) + (j)*(NZ) + (k))
#define INDEX4D_FIELDS(s, i, j, k, NX, NY, NZ) ((s)*(NX)*(NY)*(NZ) + (i)*(NY)*(NZ) + (j)*(NZ) + (k))

extern "C" {
__global__ void Back_source(
    int step, int iteration, float dx, float dy, float dz,
    const int* __restrict__ srcinfo1, const float srcinfo2, const float* __restrict__ srcwaveforms,
    float* Ex, float* Ey, float* Ez, float* uE4,
    int NX, int NY, int NZ, int NY_SRCINFO, int polarisation,int iterations
) {
    int src = blockIdx.x * blockDim.x + threadIdx.x; // 对应源维度
    int s = blockIdx.y * blockDim.y + threadIdx.y;   // 对应 step 维度
    if (src < NY_SRCINFO && s < step) {
        int index = s * (iterations * NY_SRCINFO) + iteration* NY_SRCINFO + src;
//         printf("%d %d %d-",s,iteration,src);
        // 从 srcinfo1 中获取源位置信息 (i, j, k)
        int i = srcinfo1[s * NY_SRCINFO * 3 + src * 3 + 0];
        int j = srcinfo1[s * NY_SRCINFO * 3 + src * 3 + 1];
        int k = srcinfo1[s * NY_SRCINFO * 3 + src * 3 + 2];
//         float dl = srcinfo2; // 每个源可能有不同的 dl 值
        float waveform_value = srcwaveforms[index];
//         float scale = waveform_value * dl / (dx * dy * dz);

            if (polarisation == 0) {
                Ex[INDEX4D_FIELDS(s, i, j, k, NX, NY, NZ)] -= waveform_value;//uE4[INDEX4D_FIELDS(s, i, j, k, NX, NY, NZ)] * scale;
            }
            else if (polarisation == 1) {
                Ey[INDEX4D_FIELDS(s, i, j, k, NX, NY, NZ)] -= waveform_value;//uE4[INDEX4D_FIELDS(s, i, j, k, NX, NY, NZ)] * scale;
            }
            else if (polarisation == 2){
                Ez[INDEX4D_FIELDS(s, i, j, k, NX, NY, NZ)] -= waveform_value;//uE4[INDEX4D_FIELDS(s, i, j, k, NX, NY, NZ)] * scale;
            }
        }
    }

void back_source(
    int step, int iteration, float dx, float dy, float dz,
    const int* __restrict__ srcinfo1, const float srcinfo2, const float* __restrict__ srcwaveforms,
    float* Ex, float* Ey, float* Ez, float* uE4,
    int NX, int NY, int NZ, int NY_SRCINFO, int polarisation,int iterations
) {
    dim3 threadsPerBlock(16, 16); // 每个块 16x16 线程
//     dim3 numBlocks((NY_SRCINFO + 15) / 16, (step + 15) / 16); // 按源和 step 分块
    dim3 numBlocks((NY_SRCINFO + threadsPerBlock.x - 1) / threadsPerBlock.x,
               (step + threadsPerBlock.y - 1) / threadsPerBlock.y);
    Back_source<<<numBlocks, threadsPerBlock>>>(
        step, iteration, dx, dy, dz,
        srcinfo1, srcinfo2, srcwaveforms,
        Ex, Ey, Ez, uE4, NX, NY, NZ, NY_SRCINFO, polarisation,iterations
    );
    cudaDeviceSynchronize();
}

}
