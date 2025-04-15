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


__forceinline__ __device__ void memcpy_thread(volatile char* dst, const volatile char* src, uint size)
{
        for( int i=0;i<size;i++)
                dst[i]=src[i];
}


__shared__ char int_to_char_map[10];
__device__ void init_int_to_char_map()
{
	int_to_char_map[0]='0'; int_to_char_map[1]='1'; int_to_char_map[2]='2'; int_to_char_map[3]='3'; int_to_char_map[4]='4'; int_to_char_map[5]='5'; int_to_char_map[6]='6'; int_to_char_map[7]='7'; int_to_char_map[8]='8'; int_to_char_map[9]='9';
}
	
__device__ void print_uint(char* tgt, int input, int *len){
        if (input<10) {tgt[0]=int_to_char_map[input]; tgt[1]=0; *len=1; return;}
        char count=0;
        while(input>0)
        {
                tgt[count]=int_to_char_map[input%10];
                count++;
                input/=10;
        }
        *len=count;
        count--;
        char reverse=0;
        while(count>0)
        {
                char tmp=tgt[count];
                tgt[count]=tgt[reverse];
                count--;
                tgt[reverse]=tmp;
                reverse++;
        }
}


__device__ volatile char* get_row(volatile uchar** cur_page_ptr, size_t* cur_page_offset, size_t req_file_offset, int max_file_size, int fd, int type)
{
        if (*cur_page_ptr!=NULL && *cur_page_offset+FS_BLOCKSIZE>req_file_offset)
                return (volatile char*)(*cur_page_ptr+(req_file_offset&(FS_BLOCKSIZE-1)));

        // remap
        if (*cur_page_ptr && gmunmap(*cur_page_ptr,0)) ERROR("Unmap failed");

        int mapsize=(max_file_size-req_file_offset)>FS_BLOCKSIZE?FS_BLOCKSIZE:(max_file_size-req_file_offset);

        *cur_page_offset=(req_file_offset& (~(FS_BLOCKSIZE-1)));// round to the beg. of the page
        *cur_page_ptr=(volatile uchar*) gmmap(NULL, mapsize,0,type, fd,*cur_page_offset);
        if (*cur_page_ptr == GMAP_FAILED) ERROR("MMAP failed");

        return (volatile char*)(*cur_page_ptr+(req_file_offset&(FS_BLOCKSIZE-1)));
}
struct _pagehelper{
        volatile uchar* page;
        size_t file_offset;
};

//#define alpha(src)      (((src)>=65 && (src)<=90)||( (src)>=97 && (src)<=122)|| (src)==95 || (src)==39)
#define alpha(src)      (((src)>=65 && (src)<=90)||( (src)>=97 && (src)<=122)|| (src)==95)
#define INPUT_PREFETCH_ARRAY (128*33)
#define INPUT_PREFETCH_SIZE (128*32)

#define CORPUS_PREFETCH_SIZE (16384)

__shared__ char input[INPUT_PREFETCH_ARRAY];

__shared__ char corpus[CORPUS_PREFETCH_SIZE+32+1]; // just in case we need the leftovers

__device__ int find_overlap(char* dst)
{
	__shared__ int res;
	if(threadIdx.x==0){
		res=0;
		int i=0;
		for(;i<32&&alpha(dst[i]);i++);
		res=i;
	}
	__syncthreads();
	return res;
	
}
	
		

__device__ void prefetch_banks(char *dst, volatile char *src, int data_size, int total_buf)
{
	__syncthreads();
	int i=0;

	for(i=threadIdx.x;i<data_size;i+=blockDim.x)
	{
		int offset=(i>>5)*33+(i&31);
		dst[offset]=src[i];
	}
	for(;i<total_buf;i+=blockDim.x) {
		int offset=(i>>5)*33+(i&31);
		dst[offset]=0;
	}
	__syncthreads();
}

__device__ void prefetch(char *dst, volatile char *src, int data_size, int total_buf)
{
	__syncthreads();
	int i=0;
	for(i=threadIdx.x;i<data_size;i+=blockDim.x)
	{
		dst[i]=src[i];
	}
	for(;i<total_buf;i+=blockDim.x) dst[i]=0;
	__syncthreads();
}
#define WARP_COPY(dst,src) (dst)[threadIdx.x&31]=(src)[threadIdx.x&31];
#define LEN_ZERO (-1)
#define NO_MATCH 0
#define MATCH  1


__device__ int match_string( char* a, char*data, int data_size, char* wordlen)
{
	int matches=0;
	char sizecount=0;
	char word_start=1;
	if (*a==0) return -1;
	
	for(int i=0;i<data_size;i++)
	{
		if (!alpha(data[i])) { 
			if ((sizecount == 32 || a[sizecount]=='\0' ) && word_start ) { matches++; *wordlen=sizecount;}
			word_start=1;
			sizecount=0;
		}else{

			if (a[sizecount]==data[i]) { sizecount++; }
			else {	word_start=0;	sizecount=0;}
		}
	}

	return matches;
}
__device__ int d_dbg;
__shared__ char current_db_name[FILENAME_SIZE+1];
__device__ char* get_next(char* str, char** next, int* db_strlen){
	__shared__ int beg;
	__shared__ int i;
	char db_name_ptr=0;
	if (str[0]=='\0') return NULL;
		
	BEGIN_SINGLE_THREAD
	beg=-1;
	for(i=0; (str[i]==' '||str[i]=='\t'||str[i]==','||str[i]=='\r'||str[i]=='\n');i++);
	beg=i; 
	for(;str[i]!='\n' && str[i]!='\r' && str[i]!='\0' && str[i]!=',' && i<64 ;i++,db_name_ptr++)
		current_db_name[db_name_ptr]=str[i];

	current_db_name[db_name_ptr]='\0';
	*db_strlen=i-beg;

	END_SINGLE_THREAD

	if (i-beg==64) return NULL;
	if (i-beg==0) return NULL;
	
	*next=&str[i+1];
	return current_db_name;
}

#define ROW_SIZE (128*32)
#define PREFETCH_SIZE 16384

__device__ int global_output;
__shared__ int output_count;
void __global__ vector_add(char* p_x, char* p_y, char* p_z, int nblocks, int nthreads)
{
	__shared__ int zfd_y;
	__shared__ int zfd_x;
	__shared__ int zfd_z;

#define MB (1<<20)
	
	__shared__ int toInit;
	
	zfd_x=gopen(p_x,O_GRDONLY);
	if (zfd_x<0) ERROR("Failed to open matrix");

	zfd_z=gopen(p_y,O_GWRONCE);
	if (zfd_z<0) ERROR("Failed to open output");
		
	zfd_y=gopen(p_z,O_GRDONLY);
        if (zfd_y < 0)
          ERROR("Failed to open vector");

	size_t N = fstat(zfd_x) / sizeof(float);

	volatile float* x=(volatile float*)gmmap(NULL, fstat(zfd_x),0, O_GRDONLY, zfd_x, 0);
	volatile float* y=(volatile float*)gmmap(NULL, fstat(zfd_y),0, O_GRDONLY, zfd_y, 0);
	volatile float* z=(volatile float*)gmmap(NULL, fstat(zfd_z),0, O_GRDONLY, zfd_z, 0);

	if (x==GMAP_FAILED) ERROR("GMMAP failed");
	if (y==GMAP_FAILED) ERROR("GMMAP failed");
	if (z==GMAP_FAILED) ERROR("GMMAP failed");
	
	BEGIN_SINGLE_THREAD
		toInit=init_lock.try_wait();
	
		if (toInit == 1)
		{
			single_thread_ftruncate(zfd_z,0);
			__threadfence();
			init_lock.signal();
		}
	END_SINGLE_THREAD
	int size = (N / nthreads);
	int off = threadIdx.x * size;
    for (int i = 0; i < size; ++i) {
        if (i + off > N) { break; }
		z[i + off] = x[i + off] + y[i + off];
	}
	if (gmunmap(x,0)) ERROR("Failed to unmap big matrix");
	if (gmunmap(y,0)) ERROR("Failed to unmap output");
	if (gmunmap(z,0)) ERROR("Failed to unmap vector");

	gclose(zfd_x);
	gclose(zfd_y);
	gclose(zfd_z);
}




void init_device_app(){
      CUDA_SAFE_CALL(cudaDeviceSetLimit(cudaLimitMallocHeapSize,1<<30));
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


