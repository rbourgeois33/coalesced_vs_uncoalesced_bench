# GPU Uncoalesced Bandwidth Benchmark

## 1. Benchmark Explanation

This benchmark compares the memory bandwidth (BW) of several access patterns using Kokkos:

### Coalesced Read/Write (Baseline)
```cpp
KOKKOS_LAMBDA(const _SIZE_ i) {
    write(i) = read(i);
}
```

### Uncoalesced Read
```cpp
KOKKOS_LAMBDA(const _SIZE_ i) {
    _SIZE_ rindex = indirections(i);
    write(i) = read(rindex);
}
```

### Uncoalesced Write
```cpp
KOKKOS_LAMBDA(const _SIZE_ i) {
    _SIZE_ rindex = indirections(i);
    write(rindex) = read(i);
}
```

The benchmark varies the random indirection size from `0` to the full vector size. The vector size is 2^28 doubles, which should fully utilize GPU memory bandwidth: Increasing the size beyond this point no longer improves throughput. 

---

## 2. Benchmark Results

All plots generated from the benchmark are saved in the `results/` folder. Each plot corresponds to a GPU and visualizes:

- Uncoalesced Read / Coalesced ratio
- Uncoalesced Write / Coalesced ratio  

Note: Coalesced BW exceeds H100 BW (2TB/s), i was unable to erase cache... . I suspect A100 BW is skewed too, great caches ! :)

![Nvidia V100](results/Tesla_V100-SXM2-32GB_ratios.png)
![Nvidia A100](results/NVIDIA_A100-SXM4-80GB_ratios.png)
![Nvidia H100](results/NVIDIA_H100_80GB_HBM3_ratios.png)
![AMD MI250X](results/AMD_Instinct_MI250X_ratios.png)
![AMD MI300A](results/AMD_Instinct_MI300A_ratios.png)
![Intel PVC MAX 1550](results/PVC_ratios.png)
Intel results: courtesy of Daniel Arndt !
---

## 3. Observation / Interpretation
For all architectures, uncoalesced read operations are more expensive than uncoalesced write operations. This is not surprising as it is a well known fact that read is more expensive than write. However, we observe here that the ratio gets worse as the inderection grows.

There is an exception for MI300A with the read becoming more expensive than the write for very large indirections.

The way the performance decreases with the size of the indirection is a watermark of the cache structure. We globally observe two plateau that I think correspond to L1 cache misses then L2 cache misses respectively. The compressible gas dynamics enthusiast will see the striking resemblance with shock tube profiles !

Florent Duguet @ Nvidia:
During non-coalesced read access, the various caches are populated with data, but different SMs accessing different data from the same cache line only replicate the read (from L2 cache to L1 cache). In the case of a write access, the L1 caches must be merged into the L2 cache to maintain coherence (i.e. L1 cache is write through), which requires additional processing by the memory units.

## 4. Compilation and Run Instructions

### Requirements
- Python 3 with `numpy` and `matplotlib`
- CMake
- Kokkos-enabled GPU (NVIDIA or AMD)

### Build Steps
```bash
mkdir build
cd build
cmake -DKokkos_ENABLE_CUDA=ON ..        # for NVIDIA GPUs
# OR
cmake -DKokkos_ENABLE_HIP=ON ..         # for AMD GPUs
make
```

### Run the Benchmark
```bash
./../run-bench.sh
```

The plot will be saved as `results/GPU_NAME_ratios.png`.

---
