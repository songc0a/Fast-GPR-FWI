#include <stdio.h>
#include <cuda_runtime.h>

#define INDEX2D_R(m, n,NY_R) (m)*(NY_R)+(n)
#define INDEX4D_FIELDS(s, i, j, k, NX_FIELDS, NY_FIELDS, NZ_FIELDS) \
    (s) * (NX_FIELDS * NY_FIELDS * NZ_FIELDS) + (i) * (NY_FIELDS * NZ_FIELDS) + (j) * (NZ_FIELDS) + (k)
#define INDEX4D_PHI1(p, i, j, k,NX_PHI1,NY_PHI1,NZ_PHI1) (p)*(NX_PHI1)*(NY_PHI1)*(NZ_PHI1)+(i)*(NY_PHI1)*(NZ_PHI1)+(j)*(NZ_PHI1)+(k)
#define INDEX4D_PHI2(p, i, j, k,NX_PHI2,NY_PHI2,NZ_PHI2) (p)*(NX_PHI2)*(NY_PHI2)*(NZ_PHI2)+(i)*(NY_PHI2)*(NZ_PHI2)+(j)*(NZ_PHI2)+(k)


extern "C" {

__global__ void Order1_xminus(int xs, int xf, int ys, int yf, int zs, int zf, int NX_PHI1, int NY_PHI1, int NZ_PHI1, int NX_PHI2, int NY_PHI2, int NZ_PHI2, int NY_R,   float* __restrict__ Ex,   float* __restrict__ Ey,   float* __restrict__ Ez,   float* __restrict__ Hx,  float *Hy,  float *Hz,  float *PHI1,  float *PHI2,   float* __restrict__ RA,   float* __restrict__ RB,   float* __restrict__ RE,   float* __restrict__ RF,  float d,float *updatecoeffsH, int NX_FIELDS, int NY_FIELDS, int NZ_FIELDS,int step) {

    // Obtain the linear index corresponding to the current tREad
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Convert the linear index to subscripts for PML PHI1 (4D) arrays
    int p1 = idx / (NX_PHI1 * NY_PHI1 * NZ_PHI1);
    int i1 = (idx % (NX_PHI1 * NY_PHI1 * NZ_PHI1)) / (NY_PHI1 * NZ_PHI1);
    int j1 = ((idx % (NX_PHI1 * NY_PHI1 * NZ_PHI1)) % (NY_PHI1 * NZ_PHI1)) / NZ_PHI1;
    int k1 = ((idx % (NX_PHI1 * NY_PHI1 * NZ_PHI1)) % (NY_PHI1 * NZ_PHI1)) % NZ_PHI1;

    // Convert the linear index to subscripts for PML PHI2 (4D) arrays
    int p2 = idx / (NX_PHI2 * NY_PHI2 * NZ_PHI2);
    int i2 = (idx % (NX_PHI2 * NY_PHI2 * NZ_PHI2)) / (NY_PHI2 * NZ_PHI2);
    int j2 = ((idx % (NX_PHI2 * NY_PHI2 * NZ_PHI2)) % (NY_PHI2 * NZ_PHI2)) / NZ_PHI2;
    int k2 = ((idx % (NX_PHI2 * NY_PHI2 * NZ_PHI2)) % (NY_PHI2 * NZ_PHI2)) % NZ_PHI2;

    float RA01, RB0, RE0, RF0, dEy, dEz;
    float dx = d;
    int ii, jj, kk;
    int nx = xf - xs;
    int ny = yf - ys;
    int nz = zf - zs;

    if (p1 <step && i1 < nx && j1 < ny && k1 < nz) {
        // Subscripts for field arrays
        ii = xf - (i1 + 1);
        jj = j1 + ys;
        kk = k1 + zs;

        // PML coefficients
        RA01 = RA[INDEX2D_R(0,i1,NY_R)] - 1;
        RB0 = RB[INDEX2D_R(0,i1,NY_R)];
        RE0 = RE[INDEX2D_R(0,i1,NY_R)];
        RF0 = RF[INDEX2D_R(0,i1,NY_R)];

        // Hy
         
        dEz = (Ez[INDEX4D_FIELDS(p1,ii+1,jj,kk,NX_FIELDS,NY_FIELDS,NZ_FIELDS)] - Ez[INDEX4D_FIELDS(p1,ii,jj,kk,NX_FIELDS,NY_FIELDS,NZ_FIELDS)]) / dx;
        Hy[INDEX4D_FIELDS(p1,ii,jj,kk,NX_FIELDS,NY_FIELDS,NZ_FIELDS)] = Hy[INDEX4D_FIELDS(p1,ii,jj,kk,NX_FIELDS,NY_FIELDS,NZ_FIELDS)] + updatecoeffsH[INDEX4D_FIELDS(p1,ii,jj,kk,NX_FIELDS,NY_FIELDS,NZ_FIELDS)] * (RA01 * dEz + RB0 * PHI1[INDEX4D_PHI1(p1,i1,j1,k1,NX_PHI1,NY_PHI1,NZ_PHI1)]);
        PHI1[INDEX4D_PHI1(p1,i1,j1,k1,NX_PHI1,NY_PHI1,NZ_PHI1)] = RE0 * PHI1[INDEX4D_PHI1(p1,i1,j1,k1,NX_PHI1,NY_PHI1,NZ_PHI1)] - RF0 * dEz;
    }

    if (p2 <step && i2 < nx && j2 < ny && k2 < nz) {
        // Subscripts for field arrays
        ii = xf - (i2 + 1);
        jj = j2 + ys;
        kk = k2 + zs;

        // PML coefficients
        RA01 = RA[INDEX2D_R(0,i2,NY_R)] - 1;
        RB0 = RB[INDEX2D_R(0,i2,NY_R)];
        RE0 = RE[INDEX2D_R(0,i2,NY_R)];
        RF0 = RF[INDEX2D_R(0,i2,NY_R)];

        // Hz

        dEy = (Ey[INDEX4D_FIELDS(p2,ii+1,jj,kk,NX_FIELDS,NY_FIELDS,NZ_FIELDS)] - Ey[INDEX4D_FIELDS(p2,ii,jj,kk,NX_FIELDS,NY_FIELDS,NZ_FIELDS)]) / dx;
        Hz[INDEX4D_FIELDS(p2,ii,jj,kk,NX_FIELDS,NY_FIELDS,NZ_FIELDS)] = Hz[INDEX4D_FIELDS(p2,ii,jj,kk,NX_FIELDS,NY_FIELDS,NZ_FIELDS)] - updatecoeffsH[INDEX4D_FIELDS(p2,ii,jj,kk,NX_FIELDS,NY_FIELDS,NZ_FIELDS)] * (RA01 * dEy + RB0 * PHI2[INDEX4D_PHI2(p2,i2,j2,k2,NX_PHI2,NY_PHI2,NZ_PHI2)]);
        PHI2[INDEX4D_PHI2(p2,i2,j2,k2,NX_PHI2,NY_PHI2,NZ_PHI2)] = RE0 * PHI2[INDEX4D_PHI2(p2,i2,j2,k2,NX_PHI2,NY_PHI2,NZ_PHI2)] - RF0 * dEy;
    }
}





// 接口函数：用于在 C++ 中调用 CUDA 内核函数



void order1_xminus(
    int xs, int xf, int ys, int yf, int zs, int zf,
    int NX_PHI1, int NY_PHI1, int NZ_PHI1,
    int NX_PHI2, int NY_PHI2, int NZ_PHI2,
    int NY_R,
     float* d_Ex,
     float* d_Ey,
     float* d_Ez,
     float* d_Hx,
    float* d_Hy,
    float* d_Hz,
    float* d_PHI1,
    float* d_PHI2,
     float* d_RA,
     float* d_RB,
     float* d_RE,
     float* d_RF,
    float d,
    float* d_updatecoeffsH,
    int NX_FIELDS, int NY_FIELDS, int NZ_FIELDS,int step
) {
    dim3 blockSize(256,1,1);
    dim3 gridSize(((NX_PHI1+1)*(NY_PHI1+1)*(NZ_PHI1+1)+blockSize.x-1/blockSize.x),1,1);

    // 调用 CUDA 内核函数
    Order1_xminus<<<gridSize, blockSize>>>(
        xs, xf, ys, yf, zs, zf,
        NX_PHI1, NY_PHI1, NZ_PHI1,
        NX_PHI2, NY_PHI2, NZ_PHI2,
        NY_R,
        d_Ex, d_Ey, d_Ez, d_Hx, d_Hy, d_Hz, d_PHI1, d_PHI2,
        d_RA, d_RB, d_RE, d_RF,
        d, d_updatecoeffsH, NX_FIELDS, NY_FIELDS, NZ_FIELDS,step
    );

    // 检查 CUDA 错误
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        printf("CUDA 内核启动失败: %s\n", cudaGetErrorString(error));
        return;
    }
    cudaDeviceSynchronize();
}


__global__ void Order1_xplus(int xs, int xf, int ys, int yf, int zs, int zf, int NX_PHI1, int NY_PHI1, int NZ_PHI1, int NX_PHI2, int NY_PHI2, int NZ_PHI2, int NY_R,   float* __restrict__ Ex,   float* __restrict__ Ey,   float* __restrict__ Ez,   float* __restrict__ Hx,  float *Hy,  float *Hz,  float *PHI1,  float *PHI2,   float* __restrict__ RA,   float* __restrict__ RB,   float* __restrict__ RE,   float* __restrict__ RF,  float d,float *updatecoeffsH, int NX_FIELDS, int NY_FIELDS, int NZ_FIELDS,int step) {


    // Obtain the linear index corresponding to the current tREad
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    // Convert the linear index to subscripts for PML PHI1 (4D) arrays
    int p1 = idx / (NX_PHI1 * NY_PHI1 * NZ_PHI1);
    int i1 = (idx % (NX_PHI1 * NY_PHI1 * NZ_PHI1)) / (NY_PHI1 * NZ_PHI1);
    int j1 = ((idx % (NX_PHI1 * NY_PHI1 * NZ_PHI1)) % (NY_PHI1 * NZ_PHI1)) / NZ_PHI1;
    int k1 = ((idx % (NX_PHI1 * NY_PHI1 * NZ_PHI1)) % (NY_PHI1 * NZ_PHI1)) % NZ_PHI1;

    // Convert the linear index to subscripts for PML PHI2 (4D) arrays
    int p2 = idx / (NX_PHI2 * NY_PHI2 * NZ_PHI2);
    int i2 = (idx % (NX_PHI2 * NY_PHI2 * NZ_PHI2)) / (NY_PHI2 * NZ_PHI2);
    int j2 = ((idx % (NX_PHI2 * NY_PHI2 * NZ_PHI2)) % (NY_PHI2 * NZ_PHI2)) / NZ_PHI2;
    int k2 = ((idx % (NX_PHI2 * NY_PHI2 * NZ_PHI2)) % (NY_PHI2 * NZ_PHI2)) % NZ_PHI2;

     float RA01, RB0, RE0, RF0, dEy, dEz;
     float dx = d;
    int ii, jj, kk;
    int nx = xf - xs;
    int ny = yf - ys;
    int nz = zf - zs;

    if (p1 <step && i1 < nx && j1 < ny && k1 < nz) {
        // Subscripts for field arrays
        ii = i1 + xs;
        jj = j1 + ys;
        kk = k1 + zs;

        // PML coefficients
        RA01 = RA[INDEX2D_R(0,i1,NY_R)] - 1;
        RB0 = RB[INDEX2D_R(0,i1,NY_R)];
        RE0 = RE[INDEX2D_R(0,i1,NY_R)];
        RF0 = RF[INDEX2D_R(0,i1,NY_R)];

        // Hy
         
        dEz = (Ez[INDEX4D_FIELDS(p1,ii+1,jj,kk,NX_FIELDS,NY_FIELDS,NZ_FIELDS)] - Ez[INDEX4D_FIELDS(p1,ii,jj,kk,NX_FIELDS,NY_FIELDS,NZ_FIELDS)]) / dx;
        Hy[INDEX4D_FIELDS(p1,ii,jj,kk,NX_FIELDS,NY_FIELDS,NZ_FIELDS)] = Hy[INDEX4D_FIELDS(p1,ii,jj,kk,NX_FIELDS,NY_FIELDS,NZ_FIELDS)] + updatecoeffsH[INDEX4D_FIELDS(p1,ii,jj,kk,NX_FIELDS,NY_FIELDS,NZ_FIELDS)] * (RA01 * dEz + RB0 * PHI1[INDEX4D_PHI1(p1,i1,j1,k1,NX_PHI1,NY_PHI1,NZ_PHI1)]);

        PHI1[INDEX4D_PHI1(p1,i1,j1,k1,NX_PHI1,NY_PHI1,NZ_PHI1)] = RE0 * PHI1[INDEX4D_PHI1(p1,i1,j1,k1,NX_PHI1,NY_PHI1,NZ_PHI1)] - RF0 * dEz;
    }

    if (p2 <step && i2 < nx && j2 < ny && k2 < nz) {

        // Subscripts for field arrays
        ii = i2 + xs;
        jj = j2 + ys;
        kk = k2 + zs;

        // PML coefficients
        RA01 = RA[INDEX2D_R(0,i2,NY_R)] - 1;
        RB0 = RB[INDEX2D_R(0,i2,NY_R)];
        RE0 = RE[INDEX2D_R(0,i2,NY_R)];
        RF0 = RF[INDEX2D_R(0,i2,NY_R)];

        // Hz
         
        dEy = (Ey[INDEX4D_FIELDS(p2,ii+1,jj,kk,NX_FIELDS,NY_FIELDS,NZ_FIELDS)] - Ey[INDEX4D_FIELDS(p2,ii,jj,kk,NX_FIELDS,NY_FIELDS,NZ_FIELDS)]) / dx;
        Hz[INDEX4D_FIELDS(p2,ii,jj,kk,NX_FIELDS,NY_FIELDS,NZ_FIELDS)] = Hz[INDEX4D_FIELDS(p2,ii,jj,kk,NX_FIELDS,NY_FIELDS,NZ_FIELDS)] - updatecoeffsH[INDEX4D_FIELDS(p2,ii,jj,kk,NX_FIELDS,NY_FIELDS,NZ_FIELDS)] * (RA01 * dEy + RB0 * PHI2[INDEX4D_PHI2(p2,i2,j2,k2,NX_PHI2,NY_PHI2,NZ_PHI2)]);

        PHI2[INDEX4D_PHI2(p2,i2,j2,k2,NX_PHI2,NY_PHI2,NZ_PHI2)] = RE0 * PHI2[INDEX4D_PHI2(p2,i2,j2,k2,NX_PHI2,NY_PHI2,NZ_PHI2)] - RF0 * dEy;
    }
}


void order1_xplus(
    int xs, int xf, int ys, int yf, int zs, int zf,
    int NX_PHI1, int NY_PHI1, int NZ_PHI1,
    int NX_PHI2, int NY_PHI2, int NZ_PHI2,
    int NY_R,
     float* d_Ex,               // Ex 数组
     float* d_Ey,               // Ey 数组
     float* d_Ez,               // Ez 数组
     float* d_Hx,               // Hx 数组
    float* d_Hy,                     // Hy 数组
    float* d_Hz,                     // Hz 数组
    float* d_PHI1,                   // PHI1 数组
    float* d_PHI2,                   // PHI2 数组
     float* d_RA,               // RA 数组
     float* d_RB,               // RB 数组
     float* d_RE,               // RE 数组
     float* d_RF,               // RF 数组
    float d,                         // d 参数
    float* d_updatecoeffsH,          // updatecoeffsH 数组
    int NX_FIELDS, int NY_FIELDS, int NZ_FIELDS,int step
) {
    // 配置 CUDA 的块大小和网格大小
    dim3 blockSize(256,1,1);
    dim3 gridSize(((NX_PHI1+1)*(NY_PHI1+1)*(NZ_PHI1+1)+blockSize.x-1/blockSize.x),1,1);

    // 调用 CUDA 内核函数
    Order1_xplus<<<gridSize, blockSize>>>(
        xs, xf, ys, yf, zs, zf,
        NX_PHI1, NY_PHI1, NZ_PHI1,
        NX_PHI2, NY_PHI2, NZ_PHI2,
        NY_R,
        d_Ex, d_Ey, d_Ez, d_Hx, d_Hy, d_Hz, d_PHI1, d_PHI2,
        d_RA, d_RB, d_RE, d_RF,
        d, d_updatecoeffsH, NX_FIELDS, NY_FIELDS, NZ_FIELDS,step
    );

    // 检查内核执行的错误
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        printf("CUDA 内核启动失败: %s\n", cudaGetErrorString(error));
        return;
    }

    // 使用 cudaDeviceSynchronize() 确保内核执行完成
    cudaDeviceSynchronize();

    // 再次检查同步后的错误

}



__global__ void Order1_yminus(int xs, int xf, int ys, int yf, int zs, int zf, int NX_PHI1, int NY_PHI1, int NZ_PHI1, int NX_PHI2, int NY_PHI2, int NZ_PHI2, int NY_R,    float* __restrict__ Ex,   float* __restrict__ Ey,   float* __restrict__ Ez,  float *Hx,   float* __restrict__ Hy,  float *Hz,  float *PHI1,  float *PHI2,   float* __restrict__ RA,   float* __restrict__ RB,   float* __restrict__ RE,   float* __restrict__ RF,  float d,float *updatecoeffsH, int NX_FIELDS, int NY_FIELDS, int NZ_FIELDS,int step) {

    
    // Obtain the linear index corresponding to the current tREad
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Convert the linear index to subscripts for PML PHI1 (4D) arrays
    int p1 = idx / (NX_PHI1 * NY_PHI1 * NZ_PHI1);
    int i1 = (idx % (NX_PHI1 * NY_PHI1 * NZ_PHI1)) / (NY_PHI1 * NZ_PHI1);
    int j1 = ((idx % (NX_PHI1 * NY_PHI1 * NZ_PHI1)) % (NY_PHI1 * NZ_PHI1)) / NZ_PHI1;
    int k1 = ((idx % (NX_PHI1 * NY_PHI1 * NZ_PHI1)) % (NY_PHI1 * NZ_PHI1)) % NZ_PHI1;

    // Convert the linear index to subscripts for PML PHI2 (4D) arrays
    int p2 = idx / (NX_PHI2 * NY_PHI2 * NZ_PHI2);
    int i2 = (idx % (NX_PHI2 * NY_PHI2 * NZ_PHI2)) / (NY_PHI2 * NZ_PHI2);
    int j2 = ((idx % (NX_PHI2 * NY_PHI2 * NZ_PHI2)) % (NY_PHI2 * NZ_PHI2)) / NZ_PHI2;
    int k2 = ((idx % (NX_PHI2 * NY_PHI2 * NZ_PHI2)) % (NY_PHI2 * NZ_PHI2)) % NZ_PHI2;

     float RA01, RB0, RE0, RF0, dEx, dEz;
     float dy = d;
    int ii, jj, kk;
    int nx = xf - xs;
    int ny = yf - ys;
    int nz = zf - zs;

    if (p1 <step && i1 < nx && j1 < ny && k1 < nz) {
        // Subscripts for field arrays
        ii = i1 + xs;
        jj = yf - (j1 + 1);
        kk = k1 + zs;

        // PML coefficients
        RA01 = RA[INDEX2D_R(0,j1,NY_R)] - 1;
        RB0 = RB[INDEX2D_R(0,j1,NY_R)];
        RE0 = RE[INDEX2D_R(0,j1,NY_R)];
        RF0 = RF[INDEX2D_R(0,j1,NY_R)];

        // Hx
         
        dEz = (Ez[INDEX4D_FIELDS(p1,ii,jj+1,kk,NX_FIELDS,NY_FIELDS,NZ_FIELDS)] - Ez[INDEX4D_FIELDS(p1,ii,jj,kk,NX_FIELDS,NY_FIELDS,NZ_FIELDS)]) / dy;
        Hx[INDEX4D_FIELDS(p1,ii,jj,kk,NX_FIELDS,NY_FIELDS,NZ_FIELDS)] = Hx[INDEX4D_FIELDS(p1,ii,jj,kk,NX_FIELDS,NY_FIELDS,NZ_FIELDS)] - updatecoeffsH[INDEX4D_FIELDS(p1,ii,jj,kk,NX_FIELDS,NY_FIELDS,NZ_FIELDS)] * (RA01 * dEz + RB0 * PHI1[INDEX4D_PHI1(p1,i1,j1,k1,NX_PHI1,NY_PHI1,NZ_PHI1)]);
        PHI1[INDEX4D_PHI1(p1,i1,j1,k1,NX_PHI1,NY_PHI1,NZ_PHI1)] = RE0 * PHI1[INDEX4D_PHI1(p1,i1,j1,k1,NX_PHI1,NY_PHI1,NZ_PHI1)] - RF0 * dEz;
    }

    if (p2 <step && i2 < nx && j2 < ny && k2 < nz) {
        // Subscripts for field arrays
        ii = i2 + xs;
        jj = yf - (j2 + 1);
        kk = k2 + zs;

        // PML coefficients
        RA01 = RA[INDEX2D_R(0,j2,NY_R)] - 1;
        RB0 = RB[INDEX2D_R(0,j2,NY_R)];
        RE0 = RE[INDEX2D_R(0,j2,NY_R)];
        RF0 = RF[INDEX2D_R(0,j2,NY_R)];

        // Hz
         
        dEx = (Ex[INDEX4D_FIELDS(p2,ii,jj+1,kk,NX_FIELDS,NY_FIELDS,NZ_FIELDS)] - Ex[INDEX4D_FIELDS(p2,ii,jj,kk,NX_FIELDS,NY_FIELDS,NZ_FIELDS)]) / dy;
        Hz[INDEX4D_FIELDS(p2,ii,jj,kk,NX_FIELDS,NY_FIELDS,NZ_FIELDS)] = Hz[INDEX4D_FIELDS(p2,ii,jj,kk,NX_FIELDS,NY_FIELDS,NZ_FIELDS)] + updatecoeffsH[INDEX4D_FIELDS(p2,ii,jj,kk,NX_FIELDS,NY_FIELDS,NZ_FIELDS)] * (RA01 * dEx + RB0 * PHI2[INDEX4D_PHI2(p2,i2,j2,k2,NX_PHI2,NY_PHI2,NZ_PHI2)]);
        PHI2[INDEX4D_PHI2(p2,i2,j2,k2,NX_PHI2,NY_PHI2,NZ_PHI2)] = RE0 * PHI2[INDEX4D_PHI2(p2,i2,j2,k2,NX_PHI2,NY_PHI2,NZ_PHI2)] - RF0 * dEx;
    }
}



void order1_yminus(
    int xs, int xf, int ys, int yf, int zs, int zf,
    int NX_PHI1, int NY_PHI1, int NZ_PHI1,
    int NX_PHI2, int NY_PHI2, int NZ_PHI2,
    int NY_R,
     float* d_Ex,               // Ex 数组
     float* d_Ey,               // Ey 数组
     float* d_Ez,               // Ez 数组
     float* d_Hx,               // Hx 数组
    float* d_Hy,                     // Hy 数组
    float* d_Hz,                     // Hz 数组
    float* d_PHI1,                   // PHI1 数组
    float* d_PHI2,                   // PHI2 数组
     float* d_RA,               // RA 数组
     float* d_RB,               // RB 数组
     float* d_RE,               // RE 数组
     float* d_RF,               // RF 数组
    float d,                         // d 参数
    float* d_updatecoeffsH,          // updatecoeffsH 数组
    int NX_FIELDS, int NY_FIELDS, int NZ_FIELDS,int step
) {
    // 配置 CUDA 的块大小和网格大小
    dim3 blockSize(256,1,1);
    dim3 gridSize(((NX_PHI1+1)*(NY_PHI1+1)*(NZ_PHI1+1)+blockSize.x-1/blockSize.x),1,1);

    // 调用 CUDA 内核函数
    Order1_yminus<<<gridSize, blockSize>>>(
        xs, xf, ys, yf, zs, zf,
        NX_PHI1, NY_PHI1, NZ_PHI1,
        NX_PHI2, NY_PHI2, NZ_PHI2,
        NY_R,
        d_Ex, d_Ey, d_Ez, d_Hx, d_Hy, d_Hz, d_PHI1, d_PHI2,
        d_RA, d_RB, d_RE, d_RF,
        d, d_updatecoeffsH, NX_FIELDS, NY_FIELDS, NZ_FIELDS,step
    );

    // 检查内核执行的错误
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        printf("CUDA 内核启动失败: %s\n", cudaGetErrorString(error));
        return;
    }

    // 使用 cudaDeviceSynchronize() 确保内核执行完成
    cudaDeviceSynchronize();


}

__global__ void Order1_yplus(int xs, int xf, int ys, int yf, int zs, int zf, int NX_PHI1, int NY_PHI1, int NZ_PHI1, int NX_PHI2, int NY_PHI2, int NZ_PHI2, int NY_R,    float* __restrict__ Ex,   float* __restrict__ Ey,   float* __restrict__ Ez,  float *Hx,   float* __restrict__ Hy,  float *Hz,  float *PHI1,  float *PHI2,   float* __restrict__ RA,   float* __restrict__ RB,   float* __restrict__ RE,   float* __restrict__ RF,  float d,float *updatecoeffsH, int NX_FIELDS, int NY_FIELDS, int NZ_FIELDS,int step) {

    
    // Obtain the linear index corresponding to the current tREad
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Convert the linear index to subscripts for PML PHI1 (4D) arrays
    int p1 = idx / (NX_PHI1 * NY_PHI1 * NZ_PHI1);
    int i1 = (idx % (NX_PHI1 * NY_PHI1 * NZ_PHI1)) / (NY_PHI1 * NZ_PHI1);
    int j1 = ((idx % (NX_PHI1 * NY_PHI1 * NZ_PHI1)) % (NY_PHI1 * NZ_PHI1)) / NZ_PHI1;
    int k1 = ((idx % (NX_PHI1 * NY_PHI1 * NZ_PHI1)) % (NY_PHI1 * NZ_PHI1)) % NZ_PHI1;

    // Convert the linear index to subscripts for PML PHI2 (4D) arrays
    int p2 = idx / (NX_PHI2 * NY_PHI2 * NZ_PHI2);
    int i2 = (idx % (NX_PHI2 * NY_PHI2 * NZ_PHI2)) / (NY_PHI2 * NZ_PHI2);
    int j2 = ((idx % (NX_PHI2 * NY_PHI2 * NZ_PHI2)) % (NY_PHI2 * NZ_PHI2)) / NZ_PHI2;
    int k2 = ((idx % (NX_PHI2 * NY_PHI2 * NZ_PHI2)) % (NY_PHI2 * NZ_PHI2)) % NZ_PHI2;

     float RA01, RB0, RE0, RF0, dEx, dEz;
     float dy = d;
    int ii, jj, kk;
    int nx = xf - xs;
    int ny = yf - ys;
    int nz = zf - zs;

    if (p1 <step && i1 < nx && j1 < ny && k1 < nz) {
        // Subscripts for field arrays
        ii = i1 + xs;
        jj = j1 + ys;
        kk = k1 + zs;

        // PML coefficients
        RA01 = RA[INDEX2D_R(0,j1,NY_R)] - 1;
        RB0 = RB[INDEX2D_R(0,j1,NY_R)];
        RE0 = RE[INDEX2D_R(0,j1,NY_R)];
        RF0 = RF[INDEX2D_R(0,j1,NY_R)];

        // Hx
         
        dEz = (Ez[INDEX4D_FIELDS(p1,ii,jj+1,kk,NX_FIELDS,NY_FIELDS,NZ_FIELDS)] - Ez[INDEX4D_FIELDS(p1,ii,jj,kk,NX_FIELDS,NY_FIELDS,NZ_FIELDS)]) / dy;
        Hx[INDEX4D_FIELDS(p1,ii,jj,kk,NX_FIELDS,NY_FIELDS,NZ_FIELDS)] = Hx[INDEX4D_FIELDS(p1,ii,jj,kk,NX_FIELDS,NY_FIELDS,NZ_FIELDS)] - updatecoeffsH[INDEX4D_FIELDS(p1,ii,jj,kk,NX_FIELDS,NY_FIELDS,NZ_FIELDS)] * (RA01 * dEz + RB0 * PHI1[INDEX4D_PHI1(p1,i1,j1,k1,NX_PHI1,NY_PHI1,NZ_PHI1)]);
        PHI1[INDEX4D_PHI1(p1,i1,j1,k1,NX_PHI1,NY_PHI1,NZ_PHI1)] = RE0 * PHI1[INDEX4D_PHI1(p1,i1,j1,k1,NX_PHI1,NY_PHI1,NZ_PHI1)] - RF0 * dEz;
    }

    if (p2 <step && i2 < nx && j2 < ny && k2 < nz) {

        // Subscripts for field arrays
        ii = i2 + xs;
        jj = j2 + ys;
        kk = k2 + zs;

        // PML coefficients
        RA01 = RA[INDEX2D_R(0,j2,NY_R)] - 1;
        RB0 = RB[INDEX2D_R(0,j2,NY_R)];
        RE0 = RE[INDEX2D_R(0,j2,NY_R)];
        RF0 = RF[INDEX2D_R(0,j2,NY_R)];

        // Hz
         
        dEx = (Ex[INDEX4D_FIELDS(p2,ii,jj+1,kk,NX_FIELDS,NY_FIELDS,NZ_FIELDS)] - Ex[INDEX4D_FIELDS(p2,ii,jj,kk,NX_FIELDS,NY_FIELDS,NZ_FIELDS)]) / dy;
        Hz[INDEX4D_FIELDS(p2,ii,jj,kk,NX_FIELDS,NY_FIELDS,NZ_FIELDS)] = Hz[INDEX4D_FIELDS(p2,ii,jj,kk,NX_FIELDS,NY_FIELDS,NZ_FIELDS)] + updatecoeffsH[INDEX4D_FIELDS(p2,ii,jj,kk,NX_FIELDS,NY_FIELDS,NZ_FIELDS)] * (RA01 * dEx + RB0 * PHI2[INDEX4D_PHI2(p2,i2,j2,k2,NX_PHI2,NY_PHI2,NZ_PHI2)]);
        PHI2[INDEX4D_PHI2(p2,i2,j2,k2,NX_PHI2,NY_PHI2,NZ_PHI2)] = RE0 * PHI2[INDEX4D_PHI2(p2,i2,j2,k2,NX_PHI2,NY_PHI2,NZ_PHI2)] - RF0 * dEx;
    }
}




void order1_yplus(
    int xs, int xf, int ys, int yf, int zs, int zf,
    int NX_PHI1, int NY_PHI1, int NZ_PHI1,
    int NX_PHI2, int NY_PHI2, int NZ_PHI2,
    int NY_R,
     float* d_Ex,               // Ex 数组
     float* d_Ey,               // Ey 数组
     float* d_Ez,               // Ez 数组
     float* d_Hx,               // Hx 数组
    float* d_Hy,                     // Hy 数组
    float* d_Hz,                     // Hz 数组
    float* d_PHI1,                   // PHI1 数组
    float* d_PHI2,                   // PHI2 数组
     float* d_RA,               // RA 数组
     float* d_RB,               // RB 数组
     float* d_RE,               // RE 数组
     float* d_RF,               // RF 数组
    float d,                         // d 参数
    float* d_updatecoeffsH,          // updatecoeffsH 数组
    int NX_FIELDS, int NY_FIELDS, int NZ_FIELDS,int step
) {
    // 配置 CUDA 的块大小和网格大小
    dim3 blockSize(256,1,1);
    dim3 gridSize(((NX_PHI1+1)*(NY_PHI1+1)*(NZ_PHI1+1)+blockSize.x-1/blockSize.x),1,1);

    // 调用 CUDA 内核函数
    Order1_yplus<<<gridSize, blockSize>>>(
        xs, xf, ys, yf, zs, zf,
        NX_PHI1, NY_PHI1, NZ_PHI1,
        NX_PHI2, NY_PHI2, NZ_PHI2,
        NY_R,
        d_Ex, d_Ey, d_Ez, d_Hx, d_Hy, d_Hz, d_PHI1, d_PHI2,
        d_RA, d_RB, d_RE, d_RF,
        d, d_updatecoeffsH, NX_FIELDS, NY_FIELDS, NZ_FIELDS,step
    );

    // 检查内核执行的错误
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        printf("CUDA 内核启动失败: %s\n", cudaGetErrorString(error));
        return;
    }

    // 使用 cudaDeviceSynchronize() 确保内核执行完成
    cudaDeviceSynchronize();

    // 再次检查同步后的错误

}



__global__ void Order1_zminus(int xs, int xf, int ys, int yf, int zs, int zf, int NX_PHI1, int NY_PHI1, int NZ_PHI1, int NX_PHI2, int NY_PHI2, int NZ_PHI2, int NY_R,    float* __restrict__ Ex,   float* __restrict__ Ey,   float* __restrict__ Ez,  float *Hx,  float *Hy,   float* __restrict__ Hz,  float *PHI1,  float *PHI2,   float* __restrict__ RA,   float* __restrict__ RB,   float* __restrict__ RE,   float* __restrict__ RF,  float d,float *updatecoeffsH, int NX_FIELDS, int NY_FIELDS, int NZ_FIELDS,int step) {

    // Obtain the linear index corresponding to the current tREad
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Convert the linear index to subscripts for PML PHI1 (4D) arrays
    int p1 = idx / (NX_PHI1 * NY_PHI1 * NZ_PHI1);
    int i1 = (idx % (NX_PHI1 * NY_PHI1 * NZ_PHI1)) / (NY_PHI1 * NZ_PHI1);
    int j1 = ((idx % (NX_PHI1 * NY_PHI1 * NZ_PHI1)) % (NY_PHI1 * NZ_PHI1)) / NZ_PHI1;
    int k1 = ((idx % (NX_PHI1 * NY_PHI1 * NZ_PHI1)) % (NY_PHI1 * NZ_PHI1)) % NZ_PHI1;

    // Convert the linear index to subscripts for PML PHI2 (4D) arrays
    int p2 = idx / (NX_PHI2 * NY_PHI2 * NZ_PHI2);
    int i2 = (idx % (NX_PHI2 * NY_PHI2 * NZ_PHI2)) / (NY_PHI2 * NZ_PHI2);
    int j2 = ((idx % (NX_PHI2 * NY_PHI2 * NZ_PHI2)) % (NY_PHI2 * NZ_PHI2)) / NZ_PHI2;
    int k2 = ((idx % (NX_PHI2 * NY_PHI2 * NZ_PHI2)) % (NY_PHI2 * NZ_PHI2)) % NZ_PHI2;

     float RA01, RB0, RE0, RF0, dEx, dEy;
     float dz = d;
    int ii, jj, kk;
    int nx = xf - xs;
    int ny = yf - ys;
    int nz = zf - zs;

    if (p1 <step && i1 < nx && j1 < ny && k1 < nz) {
        // Subscripts for field arrays
        ii = i1 + xs;
        jj = j1 + ys;
        kk = zf - (k1 + 1);

        // PML coefficients
        RA01 = RA[INDEX2D_R(0,k1,NY_R)] - 1;
        RB0 = RB[INDEX2D_R(0,k1,NY_R)];
        RE0 = RE[INDEX2D_R(0,k1,NY_R)];
        RF0 = RF[INDEX2D_R(0,k1,NY_R)];

        // Hx
         
        dEy = (Ey[INDEX4D_FIELDS(p1,ii,jj,kk+1,NX_FIELDS,NY_FIELDS,NZ_FIELDS)] - Ey[INDEX4D_FIELDS(p1,ii,jj,kk,NX_FIELDS,NY_FIELDS,NZ_FIELDS)]) / dz;
        Hx[INDEX4D_FIELDS(p1,ii,jj,kk,NX_FIELDS,NY_FIELDS,NZ_FIELDS)] = Hx[INDEX4D_FIELDS(p1,ii,jj,kk,NX_FIELDS,NY_FIELDS,NZ_FIELDS)] + updatecoeffsH[INDEX4D_FIELDS(p1,ii,jj,kk,NX_FIELDS,NY_FIELDS,NZ_FIELDS)] * (RA01 * dEy + RB0 * PHI1[INDEX4D_PHI1(p1,i1,j1,k1,NX_PHI1,NY_PHI1,NZ_PHI1)]);
        PHI1[INDEX4D_PHI1(p1,i1,j1,k1,NX_PHI1,NY_PHI1,NZ_PHI1)] = RE0 * PHI1[INDEX4D_PHI1(p1,i1,j1,k1,NX_PHI1,NY_PHI1,NZ_PHI1)] - RF0 * dEy;
    }

    if (p2 <step && i2 < nx && j2 < ny && k2 < nz) {

        // Subscripts for field arrays
        ii = i2 + xs;
        jj = j2 + ys;
        kk = zf - (k2 + 1);

        // PML coefficients
        RA01 = RA[INDEX2D_R(0,k2,NY_R)] - 1;
        RB0 = RB[INDEX2D_R(0,k2,NY_R)];
        RE0 = RE[INDEX2D_R(0,k2,NY_R)];
        RF0 = RF[INDEX2D_R(0,k2,NY_R)];

        // Hy
         
        dEx = (Ex[INDEX4D_FIELDS(p2,ii,jj,kk+1,NX_FIELDS,NY_FIELDS,NZ_FIELDS)] - Ex[INDEX4D_FIELDS(p2,ii,jj,kk,NX_FIELDS,NY_FIELDS,NZ_FIELDS)]) / dz;
        Hy[INDEX4D_FIELDS(p2,ii,jj,kk,NX_FIELDS,NY_FIELDS,NZ_FIELDS)] = Hy[INDEX4D_FIELDS(p2,ii,jj,kk,NX_FIELDS,NY_FIELDS,NZ_FIELDS)] - updatecoeffsH[INDEX4D_FIELDS(p2,ii,jj,kk,NX_FIELDS,NY_FIELDS,NZ_FIELDS)] * (RA01 * dEx + RB0 * PHI2[INDEX4D_PHI2(p2,i2,j2,k2,NX_PHI2,NY_PHI2,NZ_PHI2)]);
        PHI2[INDEX4D_PHI2(p2,i2,j2,k2,NX_PHI2,NY_PHI2,NZ_PHI2)] = RE0 * PHI2[INDEX4D_PHI2(p2,i2,j2,k2,NX_PHI2,NY_PHI2,NZ_PHI2)] - RF0 * dEx;
    }
}

void order1_zminus(
    int xs, int xf, int ys, int yf, int zs, int zf,
    int NX_PHI1, int NY_PHI1, int NZ_PHI1,
    int NX_PHI2, int NY_PHI2, int NZ_PHI2,
    int NY_R,
     float* d_Ex,               // Ex 数组
     float* d_Ey,               // Ey 数组
     float* d_Ez,               // Ez 数组
     float* d_Hx,               // Hx 数组
    float* d_Hy,                     // Hy 数组
    float* d_Hz,                     // Hz 数组
    float* d_PHI1,                   // PHI1 数组
    float* d_PHI2,                   // PHI2 数组
     float* d_RA,               // RA 数组
     float* d_RB,               // RB 数组
     float* d_RE,               // RE 数组
     float* d_RF,               // RF 数组
    float d,                         // d 参数
    float* d_updatecoeffsH,          // updatecoeffsH 数组
    int NX_FIELDS, int NY_FIELDS, int NZ_FIELDS,int step
) {
    // 配置 CUDA 的块大小和网格大小
    dim3 blockSize(256,1,1);
    dim3 gridSize(((NX_PHI1+1)*(NY_PHI1+1)*(NZ_PHI1+1)+blockSize.x-1/blockSize.x),1,1);

    // 调用 CUDA 内核函数
    Order1_zminus<<<gridSize, blockSize>>>(
        xs, xf, ys, yf, zs, zf,
        NX_PHI1, NY_PHI1, NZ_PHI1,
        NX_PHI2, NY_PHI2, NZ_PHI2,
        NY_R,
        d_Ex, d_Ey, d_Ez, d_Hx, d_Hy, d_Hz, d_PHI1, d_PHI2,
        d_RA, d_RB, d_RE, d_RF,
        d, d_updatecoeffsH, NX_FIELDS, NY_FIELDS, NZ_FIELDS,step
    );

    // 检查内核执行的错误
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        printf("CUDA 内核启动失败: %s\n", cudaGetErrorString(error));
        return;
    }

    // 使用 cudaDeviceSynchronize() 确保内核执行完成
    cudaDeviceSynchronize();

    // 再次检查同步后的错误

}


__global__ void Order1_zplus(int xs, int xf, int ys, int yf, int zs, int zf, int NX_PHI1, int NY_PHI1, int NZ_PHI1, int NX_PHI2, int NY_PHI2, int NZ_PHI2, int NY_R,    float* __restrict__ Ex,   float* __restrict__ Ey,   float* __restrict__ Ez,  float *Hx,  float *Hy,   float* __restrict__ Hz,  float *PHI1,  float *PHI2,   float* __restrict__ RA,   float* __restrict__ RB,   float* __restrict__ RE,   float* __restrict__ RF,  float d,float *updatecoeffsH, int NX_FIELDS, int NY_FIELDS, int NZ_FIELDS,int step) {


    // Obtain the linear index corresponding to the current tREad
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    // Convert the linear index to subscripts for PML PHI1 (4D) arrays
    int p1 = idx / (NX_PHI1 * NY_PHI1 * NZ_PHI1);
    int i1 = (idx % (NX_PHI1 * NY_PHI1 * NZ_PHI1)) / (NY_PHI1 * NZ_PHI1);
    int j1 = ((idx % (NX_PHI1 * NY_PHI1 * NZ_PHI1)) % (NY_PHI1 * NZ_PHI1)) / NZ_PHI1;
    int k1 = ((idx % (NX_PHI1 * NY_PHI1 * NZ_PHI1)) % (NY_PHI1 * NZ_PHI1)) % NZ_PHI1;

    // Convert the linear index to subscripts for PML PHI2 (4D) arrays
    int p2 = idx / (NX_PHI2 * NY_PHI2 * NZ_PHI2);
    int i2 = (idx % (NX_PHI2 * NY_PHI2 * NZ_PHI2)) / (NY_PHI2 * NZ_PHI2);
    int j2 = ((idx % (NX_PHI2 * NY_PHI2 * NZ_PHI2)) % (NY_PHI2 * NZ_PHI2)) / NZ_PHI2;
    int k2 = ((idx % (NX_PHI2 * NY_PHI2 * NZ_PHI2)) % (NY_PHI2 * NZ_PHI2)) % NZ_PHI2;

     float RA01, RB0, RE0, RF0, dEx, dEy;
     float dz = d;
    int ii, jj, kk;
    int nx = xf - xs;
    int ny = yf - ys;
    int nz = zf - zs;

    if (p1 <step && i1 < nx && j1 < ny && k1 < nz) {
        // Subscripts for field arrays
        ii = i1 + xs;
        jj = j1 + ys;
        kk = k1 + zs;

        // PML coefficients
        RA01 = RA[INDEX2D_R(0,k1,NY_R)] - 1;
        RB0 = RB[INDEX2D_R(0,k1,NY_R)];
        RE0 = RE[INDEX2D_R(0,k1,NY_R)];
        RF0 = RF[INDEX2D_R(0,k1,NY_R)];

        // Hx
         
        dEy = (Ey[INDEX4D_FIELDS(p1,ii,jj,kk+1,NX_FIELDS,NY_FIELDS,NZ_FIELDS)] - Ey[INDEX4D_FIELDS(p1,ii,jj,kk,NX_FIELDS,NY_FIELDS,NZ_FIELDS)]) / dz;
        Hx[INDEX4D_FIELDS(p1,ii,jj,kk,NX_FIELDS,NY_FIELDS,NZ_FIELDS)] = Hx[INDEX4D_FIELDS(p1,ii,jj,kk,NX_FIELDS,NY_FIELDS,NZ_FIELDS)] + updatecoeffsH[INDEX4D_FIELDS(p1,ii,jj,kk,NX_FIELDS,NY_FIELDS,NZ_FIELDS)] * (RA01 * dEy + RB0 * PHI1[INDEX4D_PHI1(p1,i1,j1,k1,NX_PHI1,NY_PHI1,NZ_PHI1)]);
        PHI1[INDEX4D_PHI1(p1,i1,j1,k1,NX_PHI1,NY_PHI1,NZ_PHI1)] = RE0 * PHI1[INDEX4D_PHI1(p1,i1,j1,k1,NX_PHI1,NY_PHI1,NZ_PHI1)] - RF0 * dEy;
    }

    if (p2 <step && i2 < nx && j2 < ny && k2 < nz) {
        // Subscripts for field arrays
        ii = i2 + xs;
        jj = j2 + ys;
        kk = k2 + zs;

        // PML coefficients
        RA01 = RA[INDEX2D_R(0,k2,NY_R)] - 1;
        RB0 = RB[INDEX2D_R(0,k2,NY_R)];
        RE0 = RE[INDEX2D_R(0,k2,NY_R)];
        RF0 = RF[INDEX2D_R(0,k2,NY_R)];

        // Hy
         
        dEx = (Ex[INDEX4D_FIELDS(p2,ii,jj,kk+1,NX_FIELDS,NY_FIELDS,NZ_FIELDS)] - Ex[INDEX4D_FIELDS(p2,ii,jj,kk,NX_FIELDS,NY_FIELDS,NZ_FIELDS)]) / dz;
        Hy[INDEX4D_FIELDS(p2,ii,jj,kk,NX_FIELDS,NY_FIELDS,NZ_FIELDS)] = Hy[INDEX4D_FIELDS(p2,ii,jj,kk,NX_FIELDS,NY_FIELDS,NZ_FIELDS)] - updatecoeffsH[INDEX4D_FIELDS(p2,ii,jj,kk,NX_FIELDS,NY_FIELDS,NZ_FIELDS)] * (RA01 * dEx + RB0 * PHI2[INDEX4D_PHI2(p2,i2,j2,k2,NX_PHI2,NY_PHI2,NZ_PHI2)]);
        PHI2[INDEX4D_PHI2(p2,i2,j2,k2,NX_PHI2,NY_PHI2,NZ_PHI2)] = RE0 * PHI2[INDEX4D_PHI2(p2,i2,j2,k2,NX_PHI2,NY_PHI2,NZ_PHI2)] - RF0 * dEx;
    }
}

void order1_zplus(
    int xs, int xf, int ys, int yf, int zs, int zf,
    int NX_PHI1, int NY_PHI1, int NZ_PHI1,
    int NX_PHI2, int NY_PHI2, int NZ_PHI2,
    int NY_R,
     float* d_Ex,               // Ex 数组
     float* d_Ey,               // Ey 数组
     float* d_Ez,               // Ez 数组
     float* d_Hx,               // Hx 数组
    float* d_Hy,                     // Hy 数组
    float* d_Hz,                     // Hz 数组
    float* d_PHI1,                   // PHI1 数组
    float* d_PHI2,                   // PHI2 数组
     float* d_RA,               // RA 数组
     float* d_RB,               // RB 数组
     float* d_RE,               // RE 数组
     float* d_RF,               // RF 数组
    float d,                         // d 参数
    float* d_updatecoeffsH,          // updatecoeffsH 数组
    int NX_FIELDS, int NY_FIELDS, int NZ_FIELDS,int step
) {
    // 配置 CUDA 的块大小和网格大小
    dim3 blockSize(256,1,1);
    dim3 gridSize(((NX_PHI1+1)*(NY_PHI1+1)*(NZ_PHI1+1)+blockSize.x-1/blockSize.x),1,1);

    // 调用 CUDA 内核函数
    Order1_zplus<<<gridSize, blockSize>>>(
        xs, xf, ys, yf, zs, zf,
        NX_PHI1, NY_PHI1, NZ_PHI1,
        NX_PHI2, NY_PHI2, NZ_PHI2,
        NY_R,
        d_Ex, d_Ey, d_Ez, d_Hx, d_Hy, d_Hz, d_PHI1, d_PHI2,
        d_RA, d_RB, d_RE, d_RF,
        d, d_updatecoeffsH, NX_FIELDS, NY_FIELDS, NZ_FIELDS,step
    );

    // 检查内核执行的错误
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        printf("CUDA 内核启动失败: %s\n", cudaGetErrorString(error));
        return;
    }

    // 使用 cudaDeviceSynchronize() 确保内核执行完成
    cudaDeviceSynchronize();

    // 再次检查同步后的错误

}



} // extern "C"
