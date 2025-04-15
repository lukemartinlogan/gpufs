#/* 
#* This expermental software is provided AS IS. 
#* Feel free to use/modify/distribute, 
#* If used, please retain this disclaimer and cite 
#* "GPUfs: Integrating a file system with GPUs", 
#* M Silberstein,B Ford,I Keidar,E Witchel
#* ASPLOS13, March 2013, Houston,USA
#*/

all: fs_gpu

CXXFLAGS=-fopenmp -g -O2


LIBGPUFSDIR=../../libgpufs/

include $(LIBGPUFSDIR)/Makefile

fs_gpu: vector_gpu.o
	$(NVCC) --relocatable-device-code=true -o fs_gpu fs.cu -I$(LIBGPUFSDIR) -I../../gpufs_con_lib.user/ $(LIBGPUFSDIR)/release/libgpufs.a

vector_gpu.o: vector_gpu.cu
	$(NVCC) -dc vector_gpu.cu -I$(LIBGPUFSDIR)

clean: 
	rm *.o fs_gpu



