const bool destroy_cache = true;

template <typename KernelType>
double run_and_time_kernel(const std::string& label, _SIZE_ size,
                           KernelType kernel, const size_t N_Repeat,
                           Kokkos::View<_TYPE_*> cache_destroyer) {
    double duration_ms = 0;

    for (size_t k = 0; k < N_Repeat; k++) {
        
        if (destroy_cache) {
            Kokkos::deep_copy(cache_destroyer, 42.0); //This erases the cache to avoid reuse between kernels
        }
        auto start = std::chrono::steady_clock::now();
        Kokkos::fence();
        Kokkos::parallel_for(label, size, kernel);
        Kokkos::fence();
        auto end = std::chrono::steady_clock::now();

        auto diff = end - start;
        duration_ms += std::chrono::duration<double, std::milli>(diff).count();
    }

    double BW = compute_BW(duration_ms, size, N_Repeat);

   std::cout << label + ": " + " BW: " + std::to_string(BW) + " GiB/s"<<std::endl;

    return BW;
}