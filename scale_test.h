//This finds the smallest vector size that uses the whole GPU.
int scale_test_(size_t total_mem){
    double BW_ref = 0;
    int N_Repeat = 20;
    int result=0;

    for (int pow_size = 20; pow_size<31; pow_size++){
        
        int size = 1;
        for (int k = 0; k < pow_size; k++) {
            size *= 2;
        }

        Kokkos::View<_TYPE_*> read("read", size);
        Kokkos::View<_TYPE_*> write("write", size);
        Kokkos::View<_TYPE_*> cache_destroyer(
            "cache_destroyer", total_mem / (10 * sizeof(double)));

        double BW_coalesced = run_and_time_kernel(
                "coalesced R/W", size,
                KOKKOS_LAMBDA(const _SIZE_ i) { write(i) = read(i); }, N_Repeat,
                cache_destroyer);

        if (BW_coalesced*0.99>BW_ref){
            BW_ref = BW_coalesced;
        }
        else{
            result = pow_size-1;
            break;
        }

    }
    std::cout<<"Memory bound with 2^"<<result<<"elems \n";
    Kokkos::finalize();
    return result;
}