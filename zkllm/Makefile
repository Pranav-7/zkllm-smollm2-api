# Compilers
NVCC := /usr/local/cuda/bin/nvcc

# Include and library paths
INCLUDES := -I/usr/local/cuda/include
LIBS := -L/usr/local/cuda/lib64

# get compute capability from retrieved value
ARCH := sm_90

# NVCC compiler flags
NVCC_FLAGS := -arch=$(ARCH) -std=c++17

# Source and object files
CU_SRCS := bls12-381.cu ioutils.cu commitment.cu fr-tensor.cu g1-tensor.cu proof.cu zkrelu.cu zkfc.cu tlookup.cu polynomial.cu zksoftmax.cu rescaling.cu
CU_OBJS := $(CU_SRCS:.cu=.o)
CPP_SRCS := $(wildcard *.cpp)
CPP_OBJS := $(CPP_SRCS:.cpp=.o)
TARGETS := main ppgen commit-param self-attn ffn rmsnorm skip-connection
TARGET_OBJS := $(TARGETS:=.o)

# Pattern rule for CUDA source files
%.o: %.cu
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) -dc -dlto $< -o $@

# Pattern rule for C++ source files
%.o: %.cpp
	$(NVCC) -x cu $(NVCC_FLAGS) $(INCLUDES) -dc -dlto $< -o $@

# General rule for building each target
$(TARGETS): % : %.o $(CU_OBJS) $(CPP_OBJS)
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) $(LIBS) $^ -o $@ -dlto

# Clean rule
clean:
	rm -f $(TARGET_OBJS) $(CU_OBJS) $(CPP_OBJS) $(TARGETS)

# Default rule
all: $(TARGETS)