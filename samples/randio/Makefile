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

fs_gpu: randio_gpu.o
	$(NVCC) --relocatable-device-code=true -o fs_gpu fs.cu -I$(LIBGPUFSDIR) -I../../gpufs_con_lib.user/ $(LIBGPUFSDIR)/release/libgpufs.a

randio_gpu.o: randio_gpu.cu
	$(NVCC) -dc randio_gpu.cu -I$(LIBGPUFSDIR)

clean: 
	rm *.o fs_gpu



