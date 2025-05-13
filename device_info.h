#ifdef KOKKOS_ENABLE_CUDA
#include <cuda_runtime.h>
#endif

#ifdef KOKKOS_ENABLE_HIP
#include <hip/hip_runtime.h>
#endif

std::string replace_spaces_with_underscores(std::string input) {
    for (char& c : input) {
      if (c == ' ') {
        c = '_';
      }
    }
    return input;
  }


std::string get_device_name() {
    int device;
    
    #ifdef KOKKOS_ENABLE_CUDA
    cudaGetDevice(&device);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);
    #endif
  
    #ifdef KOKKOS_ENABLE_HIP
    hipGetDevice(&device);
    hipDeviceProp_t prop;
    hipGetDeviceProperties(&prop, device);
    #endif
  
    return replace_spaces_with_underscores(prop.name);
  }
  
  
  void get_device_memory_info(size_t & free_mem, size_t & total_mem) {
    int device = 0;
  
    #ifdef KOKKOS_ENABLE_CUDA
    cudaGetDevice(&device);
    cudaMemGetInfo(&free_mem, &total_mem);
    #endif
  
    #ifdef KOKKOS_ENABLE_HIP
    hipGetDevice(&device);
    hipMemGetInfo(&free_mem, &total_mem);
    #endif
  }