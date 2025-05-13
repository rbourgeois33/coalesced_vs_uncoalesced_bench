#include <Kokkos_Core.hpp>
#include <Kokkos_Random.hpp>
#include <chrono>
#include <fstream>
#include <iostream>
#include <string>
#include <filesystem>  // C++17 and later

// Data type and indexing type
using _TYPE_ = double;
using _SIZE_ = int;

#include "device_info.h"
#include "quantities.h"
#include "run_and_time_kernel.h"
#include "scale_test.h"
#include "compute_rindex.h"


int main(int argc, char* argv[]) {

    _SIZE_ size = 0;
    size_t N_Repeat = 0;
    _SIZE_ max_dist = 0;
    _SIZE_ pow_size = 0;
    _SIZE_ pow_max_dist = 0;
    bool scale_test = false;

    Kokkos::initialize(argc, argv);
    size_t free_mem, total_mem;
    get_device_memory_info(free_mem, total_mem);


    // Parse command-line arguments
    for (int i = 1; i < argc; ++i) {

        //Size of the vectors
        if (std::string(argv[i]) == "--pow_size" && i + 1 < argc) {
            pow_size = std::stoi(argv[++i]);
            size = 1;
            for (int k = 0; k < pow_size; k++) {
                size *= 2;
            }
        //Number of kernel repetitions
        } else if (std::string(argv[i]) == "--N_Repeat" && i + 1 < argc) {
            N_Repeat = std::stoi(argv[++i]);
        
        //Size of the indirections
        } else if (std::string(argv[i]) == "--pow_max_dist" && i + 1 < argc) {
            pow_max_dist = std::stoi(argv[++i]);
            max_dist = 1;
            for (int k = 0; k < pow_max_dist; k++) {
                max_dist *= 2;
            }
        //Do we do the scale test (check theminimum size that uses the whole GPU)
        } else if (std::string(argv[i]) == "--scale_test") {
            scale_test = true;
            if (scale_test){return scale_test_(total_mem);}

        } else {
            std::cerr << "Unknown argument: " << argv[i] << std::endl;
            std::cerr << "Usage: " << argv[0]
                      << " [--pow_size <size> (2^pow_size)] [--N_Repeat <N_Repeat>] [--pow_max_dist "
                     "<pow_max_dist> (2^max-dist) optional]"
                      << std::endl;
            return EXIT_FAILURE;
        }
    }

    if (max_dist == 0) {
        std::cout << "max_dist not given, taking size "<<std::endl;
        max_dist = size;
    }

    {
        Kokkos::View<_TYPE_*> read("read", size);
        Kokkos::View<_TYPE_*> write("write", size);

        Kokkos::View<_SIZE_*> indirections("indirections", size);
        //Used between kernel to erase the cache
        Kokkos::View<_TYPE_*> cache_destroyer("cache_destroyer", total_mem / (10 * sizeof(double)));

        auto time_seed = std::chrono::high_resolution_clock::now().time_since_epoch().count();
        Kokkos::Random_XorShift64_Pool<> random_pool(time_seed);

        Kokkos::parallel_for(
            "fill indirections", size, KOKKOS_LAMBDA(const _SIZE_ i) {
                auto generator = random_pool.get_state();
                _SIZE_ rindex = compute_rindex(max_dist, size, i, generator);
                indirections(i) = rindex;
                random_pool.free_state(generator);
            });

        double BW_coalesced = run_and_time_kernel(
            "coalesced R/W", size,
            KOKKOS_LAMBDA(const _SIZE_ i) { write(i) = read(i); }, N_Repeat,
            cache_destroyer);

        double BW_uncoalesced_read = run_and_time_kernel(
            "uncoalesced read with table", size,
            KOKKOS_LAMBDA(const _SIZE_ i) {
                _SIZE_ rindex = indirections(i);
                write(i) = read(rindex);
            },
            N_Repeat, cache_destroyer);

        // run_and_time_kernel(
        //     "uncoalesced read generated on the fly", size,
        //    KOKKOS_LAMBDA(const _SIZE_ i) {
        //         auto generator = random_pool.get_state();
        //         _SIZE_ rindex = compute_rindex(max_dist, size, i, generator);
        //         write(i) = read(rindex);
        //         random_pool.free_state(generator);
        //     }, N_Repeat);

        double BW_uncoalesced_write = run_and_time_kernel(
            "uncoalesced write with table", size,
            KOKKOS_LAMBDA(const _SIZE_ i) {
                _SIZE_ rindex = indirections(i);
                write(rindex) = read(i);
            },
            N_Repeat, cache_destroyer);

        // run_and_time_kernel(
        //     "uncoalesced write generated on the fly", size,
        //     KOKKOS_LAMBDA(const _SIZE_ i) {
        //         auto generator = random_pool.get_state();
        //         _SIZE_ rindex = compute_rindex(max_dist, size, i, generator);
        //         write(rindex) = read(i);
        //         random_pool.free_state(generator);
        //     }, N_Repeat);
        
        //Print and output perfs
        double read_penalty = 1.0 - (BW_uncoalesced_read / BW_coalesced);
        double write_penalty = 1.0 - (BW_uncoalesced_write / BW_coalesced);

        std::cout << "uncoalesced read penalty = " << read_penalty << ""<<std::endl;
        std::cout << "uncoalesced write penalty = " << write_penalty << ""<<std::endl;
        std::cout << "ratio = " << BW_uncoalesced_write / BW_uncoalesced_read
                  << ""<<std::endl;
        
        std::ofstream file;
        
        auto filename = get_device_name()+".txt";
       
        bool file_exists = std::filesystem::exists(filename);
        file.open(filename, std::ios::out | std::ios::app);  // append mode
        
          // If the file is newly created, write a header or special line
        if (!file_exists) {
            file << "#GPU: "<< get_device_name()<<"\n";
            file << "Total mem:" << total_mem/GiB<<" GiB\n";
        }

        file << pow_size <<" " <<pow_max_dist << " "<<BW_coalesced << " " << BW_uncoalesced_read << " "<< BW_uncoalesced_write << std::endl;
      
        file.close();

    }

    Kokkos::finalize();

    return 0;
}
