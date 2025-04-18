/* 
* This expermental software is provided AS IS. 
* Feel free to use/modify/distribute, 
* If used, please retain this disclaimer and cite 
* "GPUfs: Integrating a file system with GPUs", 
* M Silberstein,B Ford,I Keidar,E Witchel
* ASPLOS13, March 2013, Houston,USA
*/

#include "fs_constants.h"
#include "util.cu.h"
#include "fs_calls.cu.h"
#include <sys/mman.h>
#include <stdio.h>


__device__ volatile INIT_LOCK init_lock;
__device__ volatile LAST_SEMAPHORE last_lock;

__device__ size_t random_uint(size_t state, size_t upper) {
	state = (state * 9301 + 49297) % 233280;
	float rnd = state / (float)233280.0;
	state = rnd * upper;
  return state;
}

__device__ size_t globalId() {
	size_t idx_x = blockIdx.x * blockDim.x + threadIdx.x;
 	size_t idx_y = blockIdx.y * blockDim.y + threadIdx.y;
 	size_t idx_z = blockIdx.z * blockDim.z + threadIdx.z;
 	size_t idx = idx_z * gridDim.y * blockDim.y * gridDim.x * blockDim.x + idx_y * gridDim.x * blockDim.x + idx_x; // 1D global index from 3D grid
	return idx;
}

void __global__ randio(char* p_x, int nblocks, int nthreads)
{
	__shared__ int zfd_x;

#define MB (1<<20)
	
	__shared__ int toInit;
	
	zfd_x=gopen(p_x,O_GRDONLY);
	if (zfd_x<0) ERROR("Failed to open matrix");

	size_t npages = fstat(zfd_x) / FS_BLOCKSIZE;

	volatile float* x=(volatile float*)gmmap(NULL, fstat(zfd_x),0, O_GRDONLY, zfd_x, 0);
	
	if (x==GMAP_FAILED) ERROR("GMMAP failed");
		
	BEGIN_SINGLE_THREAD
		toInit=init_lock.try_wait();
	
		if (toInit == 1)
		{
			single_thread_ftruncate(zfd_x,0);
			__threadfence();
			init_lock.signal();
		}
	END_SINGLE_THREAD
	int size = (npages / (nthreads * nblocks));
	if (size == 0) {
		size = 1;
	}
	int id = globalId() * size;
    for (int i = 0; i < size; ++i) { 
		int page = random_uint(id + i, npages);
		int off = page * FS_BLOCKSIZE;
		size_t sum = 0;
		for (int j = 0; j < FS_BLOCKSIZE; ++j) {
			sum += x[off + j];
		}
	}
	if (gmunmap(x,0)) ERROR("Failed to unmap big matrix");

	gclose(zfd_x);
}


void init_device_app(){
      CUDA_SAFE_CALL(cudaDeviceSetLimit(cudaLimitMallocHeapSize,1 * (1<<30)));
}
void init_app()
{
        // INITI LOCK   
        void* inited;

        CUDA_SAFE_CALL(cudaGetSymbolAddress(&inited,init_lock));
        CUDA_SAFE_CALL(cudaMemset(inited,0,sizeof(INIT_LOCK)));

        CUDA_SAFE_CALL(cudaGetSymbolAddress(&inited,last_lock));
        CUDA_SAFE_CALL(cudaMemset(inited,0,sizeof(LAST_SEMAPHORE)));
}

double post_app(double total_time, float trials )
{
        return 0;
        //return  sizeof(float)*VEC_FLOAT*((double)VEC_FLOAT)*2/ (total_time/trials);
}


