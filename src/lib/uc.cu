#include <cuda_runtime.h>
#include <stdio.h>

#define INDEX3D_FIELDS(i, j, k,NY_FIELDS,NZ_FIELDS) (i)*(NY_FIELDS)*(NZ_FIELDS)+(j)*(NZ_FIELDS)+(k)


extern "C" {
__constant__ float e0 = 8.8541878128e-12;
__constant__ float m0 = 1.25663706212e-06;

__global__ void ucget(float *er,float *se,float *mr,float *uE0, float *uE1, float *uE4,float *uH0, float *uH1, float *uH4,int NX_FIELDS, int NY_FIELDS, int NZ_FIELDS,float dt,float dx) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    // Convert the linear index to subscripts for 3D field arrays
    int i = idx / (NY_FIELDS * NZ_FIELDS);
    int j = (idx % (NY_FIELDS * NZ_FIELDS)) / NZ_FIELDS;
    int k = (idx % (NY_FIELDS * NZ_FIELDS)) % NZ_FIELDS;

    if (i < (NX_FIELDS-1) && j < (NY_FIELDS-1) && k < (NZ_FIELDS-1) ) {
    //atomicAdd(&run_count, 1);  // 使用原子操作增加计数器
        float HA = m0 * mr[INDEX3D_FIELDS(i, j, k,NY_FIELDS,NZ_FIELDS)] / dt;
        uH0[INDEX3D_FIELDS(i, j, k,NY_FIELDS,NZ_FIELDS)] = 1;
        uH1[INDEX3D_FIELDS(i, j, k,NY_FIELDS,NZ_FIELDS)] = (1 / dx) * 1 / HA;
        uH4[INDEX3D_FIELDS(i, j, k,NY_FIELDS,NZ_FIELDS)] = 1 / HA;
        //printf("---%f %f %f---",   uH0[INDEX3D_FIELDS(i, j, k,NY_FIELDS,NZ_FIELDS)],uH1[INDEX3D_FIELDS(i, j, k,NY_FIELDS,NZ_FIELDS)],uH4[INDEX3D_FIELDS(i, j, k,NY_FIELDS,NZ_FIELDS)]);

        //printf("--%i %i %i--",i,j,k);
        if (se[INDEX3D_FIELDS(i, j, k,NY_FIELDS,NZ_FIELDS)] > 100) {
            uE0[INDEX3D_FIELDS(i, j, k,NY_FIELDS,NZ_FIELDS)] = 0;
            uE1[INDEX3D_FIELDS(i, j, k,NY_FIELDS,NZ_FIELDS)] = 0;
            uE4[INDEX3D_FIELDS(i, j, k,NY_FIELDS,NZ_FIELDS)] = 0;
            //printf("xx%f %f %f - ",uE0[INDEX3D_FIELDS(i, j, k,NY_FIELDS,NZ_FIELDS)],uE1[INDEX3D_FIELDS(i, j, k,NY_FIELDS,NZ_FIELDS)],uE4[INDEX3D_FIELDS(i, j, k,NY_FIELDS,NZ_FIELDS)] );
        } else {
            float EA = (e0 * er[INDEX3D_FIELDS(i, j, k,NY_FIELDS,NZ_FIELDS)] / dt) + 0.5 * se[INDEX3D_FIELDS(i, j, k,NY_FIELDS,NZ_FIELDS)];
            float EB = (e0 * er[INDEX3D_FIELDS(i, j, k,NY_FIELDS,NZ_FIELDS)] / dt) - 0.5 * se[INDEX3D_FIELDS(i, j, k,NY_FIELDS,NZ_FIELDS)];
            uE0[INDEX3D_FIELDS(i, j, k,NY_FIELDS,NZ_FIELDS)] = EB / EA;
            uE1[INDEX3D_FIELDS(i, j, k,NY_FIELDS,NZ_FIELDS)] = (1 / dx) * 1 / EA;
            uE4[INDEX3D_FIELDS(i, j, k,NY_FIELDS,NZ_FIELDS)] = 1 / EA;

            //if(isnan(uE0[INDEX3D_FIELDS(i, j, k,NY_FIELDS,NZ_FIELDS)]))
           // printf("---%f %f %f---",   uE0[INDEX3D_FIELDS(i, j, k,NY_FIELDS,NZ_FIELDS)],uE1[INDEX3D_FIELDS(i, j, k,NY_FIELDS,NZ_FIELDS)],uE4[INDEX3D_FIELDS(i, j, k,NY_FIELDS,NZ_FIELDS)]);
              //printf("yy%e- ",dt);
        }
    }
}


void Ucget(float *er,float *se,float *mr,float *uE0, float *uE1, float *uE4,float *uH0, float *uH1, float *uH4,int NX_FIELDS, int NY_FIELDS, int NZ_FIELDS,float dt,float dx) {

    dim3 threads_per_block(256);
    dim3 blocks_per_grid((NX_FIELDS * NY_FIELDS * NZ_FIELDS + threads_per_block.x - 1) / threads_per_block.x);

    ucget<<<blocks_per_grid, threads_per_block>>>(er,se,mr,uE0, uE1, uE4,uH0, uH1,uH4,NX_FIELDS, NY_FIELDS, NZ_FIELDS,dt,dx);
    cudaDeviceSynchronize();
}




__global__ void ucgeta(float *er,float *se,float *mr,float *uE0, float *uE1, float *uE4,float *uH0, float *uH1, float *uH4,int NX_FIELDS, int NY_FIELDS, int NZ_FIELDS,float dt,float dx) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    // Convert the linear index to subscripts for 3D field arrays
    int i = idx / (NY_FIELDS * NZ_FIELDS);
    int j = (idx % (NY_FIELDS * NZ_FIELDS)) / NZ_FIELDS;
    int k = (idx % (NY_FIELDS * NZ_FIELDS)) % NZ_FIELDS;

    if (i < (NX_FIELDS-1) && j < (NY_FIELDS-1) && k < (NZ_FIELDS-1) ) {
    //atomicAdd(&run_count, 1);  // 使用原子操作增加计数器
        float HA = m0 * mr[INDEX3D_FIELDS(i, j, k,NY_FIELDS,NZ_FIELDS)] / dt;
        uH0[INDEX3D_FIELDS(i, j, k,NY_FIELDS,NZ_FIELDS)] = 1;
        uH1[INDEX3D_FIELDS(i, j, k,NY_FIELDS,NZ_FIELDS)] = (1 / dx) * 1 / HA;
        uH4[INDEX3D_FIELDS(i, j, k,NY_FIELDS,NZ_FIELDS)] = 1 / HA;
        //printf("---%f %f %f---",   uH0[INDEX3D_FIELDS(i, j, k,NY_FIELDS,NZ_FIELDS)],uH1[INDEX3D_FIELDS(i, j, k,NY_FIELDS,NZ_FIELDS)],uH4[INDEX3D_FIELDS(i, j, k,NY_FIELDS,NZ_FIELDS)]);

        //printf("--%i %i %i--",i,j,k);
        if (se[INDEX3D_FIELDS(i, j, k,NY_FIELDS,NZ_FIELDS)] > 100) {
            uE0[INDEX3D_FIELDS(i, j, k,NY_FIELDS,NZ_FIELDS)] = 0;
            uE1[INDEX3D_FIELDS(i, j, k,NY_FIELDS,NZ_FIELDS)] = 0;
            uE4[INDEX3D_FIELDS(i, j, k,NY_FIELDS,NZ_FIELDS)] = 0;
            //printf("xx%f %f %f - ",uE0[INDEX3D_FIELDS(i, j, k,NY_FIELDS,NZ_FIELDS)],uE1[INDEX3D_FIELDS(i, j, k,NY_FIELDS,NZ_FIELDS)],uE4[INDEX3D_FIELDS(i, j, k,NY_FIELDS,NZ_FIELDS)] );
        } else {
            float EA = (e0 * er[INDEX3D_FIELDS(i, j, k,NY_FIELDS,NZ_FIELDS)] / dt) + 0.5 * se[INDEX3D_FIELDS(i, j, k,NY_FIELDS,NZ_FIELDS)];
//             float EB = (e0 * er[INDEX3D_FIELDS(i, j, k,NY_FIELDS,NZ_FIELDS)] / dt) - 0.5 * se[INDEX3D_FIELDS(i, j, k,NY_FIELDS,NZ_FIELDS)];
            uE0[INDEX3D_FIELDS(i, j, k,NY_FIELDS,NZ_FIELDS)] =(2*e0 * er[INDEX3D_FIELDS(i, j, k,NY_FIELDS,NZ_FIELDS)])/(2*e0 * er[INDEX3D_FIELDS(i, j, k,NY_FIELDS,NZ_FIELDS)]+se[INDEX3D_FIELDS(i, j, k,NY_FIELDS,NZ_FIELDS)]*dt);
            uE1[INDEX3D_FIELDS(i, j, k,NY_FIELDS,NZ_FIELDS)] = (1 / dx) * 1 / EA;
            uE4[INDEX3D_FIELDS(i, j, k,NY_FIELDS,NZ_FIELDS)] = 1 / EA;

        }
    }
}


void Ucgeta(float *er,float *se,float *mr,float *uE0, float *uE1, float *uE4,float *uH0, float *uH1, float *uH4,int NX_FIELDS, int NY_FIELDS, int NZ_FIELDS,float dt,float dx) {

    dim3 threads_per_block(256);
    dim3 blocks_per_grid((NX_FIELDS * NY_FIELDS * NZ_FIELDS + threads_per_block.x - 1) / threads_per_block.x);

    ucgeta<<<blocks_per_grid, threads_per_block>>>(er,se,mr,uE0, uE1, uE4,uH0, uH1,uH4,NX_FIELDS, NY_FIELDS, NZ_FIELDS,dt,dx);
    cudaDeviceSynchronize();
}


}