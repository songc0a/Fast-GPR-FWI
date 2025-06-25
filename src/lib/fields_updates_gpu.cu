#include <cuda_runtime.h>
#include <stdio.h>

extern "C" {

#define INDEX4D_FIELDS(s, i, j, k, NX, NY, NZ) ((s)*(NX)*(NY)*(NZ) + (i)*(NY)*(NZ) + (j)*(NZ) + (k))

__global__ void e_fields_updates_gpu(
    float *uE0, float *uE1, float *Ex, float *Ey, float *Ez,
    float *Hx, float *Hy, float *Hz,
    int step, int NX_FIELDS, int NY_FIELDS, int NZ_FIELDS
) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_cells_per_step = NX_FIELDS * NY_FIELDS * NZ_FIELDS;
    int s = idx / total_cells_per_step;
    int i = idx % total_cells_per_step / (NY_FIELDS * NZ_FIELDS);
    int j = (idx % total_cells_per_step % (NY_FIELDS * NZ_FIELDS)) / NZ_FIELDS;
    int k = idx % total_cells_per_step % NZ_FIELDS;

    if (s < step && i > 0 && i < NX_FIELDS && j > 0 && j < NY_FIELDS && k >= 0 && k < NZ_FIELDS) {
        // 更新电场 Ez
        Ez[INDEX4D_FIELDS(s, i, j, k, NX_FIELDS, NY_FIELDS, NZ_FIELDS)] =
            uE0[INDEX4D_FIELDS(s, i, j, k, NX_FIELDS, NY_FIELDS, NZ_FIELDS)] *
            Ez[INDEX4D_FIELDS(s, i, j, k, NX_FIELDS, NY_FIELDS, NZ_FIELDS)] +
            uE1[INDEX4D_FIELDS(s, i, j, k, NX_FIELDS, NY_FIELDS, NZ_FIELDS)] *
            (Hy[INDEX4D_FIELDS(s, i, j, k, NX_FIELDS, NY_FIELDS, NZ_FIELDS)] -
             Hy[INDEX4D_FIELDS(s, i - 1, j, k, NX_FIELDS, NY_FIELDS, NZ_FIELDS)]) -
            uE1[INDEX4D_FIELDS(s, i, j, k, NX_FIELDS, NY_FIELDS, NZ_FIELDS)] *
            (Hx[INDEX4D_FIELDS(s, i, j, k, NX_FIELDS, NY_FIELDS, NZ_FIELDS)] -
             Hx[INDEX4D_FIELDS(s, i, j - 1, k, NX_FIELDS, NY_FIELDS, NZ_FIELDS)]);
    }
}

void e_fields_updates(float *uE0, float *uE1,float *Ex,float *Ey,float *Ez,float *Hx,float *Hy,float *Hz,int step, int NX_FIELDS, int NY_FIELDS, int NZ_FIELDS) {

    int total_cells = step * NX_FIELDS * NY_FIELDS * NZ_FIELDS;
    dim3 threadsPerBlock(256);
    dim3 numBlocks((total_cells + threadsPerBlock.x - 1) / threadsPerBlock.x);

    e_fields_updates_gpu<<<numBlocks, threadsPerBlock>>>(
        uE0, uE1, Ex, Ey, Ez, Hx, Hy, Hz, step, NX_FIELDS, NY_FIELDS, NZ_FIELDS);
}



__global__ void h_fields_updates_gpu(
    float *uH0, float *uH1, float *Ex, float *Ey, float *Ez,
    float *Hx, float *Hy, float *Hz,
    int step, int NX_FIELDS, int NY_FIELDS, int NZ_FIELDS
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_cells_per_step = NX_FIELDS * NY_FIELDS * NZ_FIELDS;
    int s = idx / total_cells_per_step;
    int i = idx % total_cells_per_step / (NY_FIELDS * NZ_FIELDS);
    int j = (idx % total_cells_per_step % (NY_FIELDS * NZ_FIELDS)) / NZ_FIELDS;
    int k = idx % total_cells_per_step % NZ_FIELDS;

    if (s < step && i > 0 && i < NX_FIELDS && j >= 0 && j < NY_FIELDS && k >= 0 && k < NZ_FIELDS) {
        // 更新 Hx 分量
        Hx[INDEX4D_FIELDS(s, i, j, k, NX_FIELDS, NY_FIELDS, NZ_FIELDS)] =
            uH0[INDEX4D_FIELDS(s, i, j, k, NX_FIELDS, NY_FIELDS, NZ_FIELDS)] *
            Hx[INDEX4D_FIELDS(s, i, j, k, NX_FIELDS, NY_FIELDS, NZ_FIELDS)] -
            uH1[INDEX4D_FIELDS(s, i, j, k, NX_FIELDS, NY_FIELDS, NZ_FIELDS)] *
            (Ez[INDEX4D_FIELDS(s, i, j + 1, k, NX_FIELDS, NY_FIELDS, NZ_FIELDS)] -
             Ez[INDEX4D_FIELDS(s, i, j, k, NX_FIELDS, NY_FIELDS, NZ_FIELDS)]);
    }

    if (s < step && i >= 0 && i < NX_FIELDS && j > 0 && j < NY_FIELDS && k >= 0 && k < NZ_FIELDS) {
        // 更新 Hy 分量
        Hy[INDEX4D_FIELDS(s, i, j, k, NX_FIELDS, NY_FIELDS, NZ_FIELDS)] =
            uH0[INDEX4D_FIELDS(s, i, j, k, NX_FIELDS, NY_FIELDS, NZ_FIELDS)] *
            Hy[INDEX4D_FIELDS(s, i, j, k, NX_FIELDS, NY_FIELDS, NZ_FIELDS)] +
            uH1[INDEX4D_FIELDS(s, i, j, k, NX_FIELDS, NY_FIELDS, NZ_FIELDS)] *
            (Ez[INDEX4D_FIELDS(s, i + 1, j, k, NX_FIELDS, NY_FIELDS, NZ_FIELDS)] -
             Ez[INDEX4D_FIELDS(s, i, j, k, NX_FIELDS, NY_FIELDS, NZ_FIELDS)]);
    }
}

void h_fields_updates(float *uH0, float *uH1, float *Ex,float *Ey,float *Ez,float *Hx,float *Hy,float *Hz, int step, int NX_FIELDS, int NY_FIELDS, int NZ_FIELDS) {

    int total_cells = step * NX_FIELDS * NY_FIELDS * NZ_FIELDS;
    dim3 threadsPerBlock(256);
    dim3 numBlocks((total_cells + threadsPerBlock.x - 1) / threadsPerBlock.x);

    // 启动核函数
    h_fields_updates_gpu<<<numBlocks, threadsPerBlock>>>(
        uH0, uH1, Ex, Ey, Ez, Hx, Hy, Hz, step, NX_FIELDS, NY_FIELDS, NZ_FIELDS);

}


__global__ void compute_gradients_gpu(
    float *Ez, float *Ez_last, float *Hx, float *Hy,
    float *dEz_deps, float *dEz_dsig,
    float *er, float *se,
    int step, int NX, int NY, int NZ,
    float dt, float dx
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_cells_per_step = NX * NY * NZ;
    int s = idx / total_cells_per_step;
    int i = idx % total_cells_per_step / (NY * NZ);
    int j = (idx % total_cells_per_step % (NY * NZ)) / NZ;
    int k = idx % total_cells_per_step % NZ;

    if (s >= 0 && s < step && i > 0 && i < NX && j > 0 && j < NY && k >= 0 && k < NZ) {
        int current4D = s * NX * NY * NZ + i * NY * NZ + j * NZ + k;
        dEz_deps[current4D] = (Ez[current4D]-Ez_last[current4D])/dt;
        dEz_dsig[current4D] = Ez[current4D];
    }
}

void LCG(
    float *Ez, float *Ez_last,
    float *Hx, float *Hy,
    float *dEz_deps, float *dEz_dsig,
    float *er, float *se,
    int step, int NX, int NY, int NZ,
    float dt, float dx
) {
    int total_cells = step * NX * NY * NZ;
    dim3 threadsPerBlock(256);
    dim3 numBlocks((total_cells + threadsPerBlock.x - 1) / threadsPerBlock.x);

    // 启动kernel
    compute_gradients_gpu<<<numBlocks, threadsPerBlock>>>(
        Ez, Ez_last, Hx, Hy,dEz_deps, dEz_dsig,er, se,
        step, NX, NY, NZ,dt, dx
    );
    }
}