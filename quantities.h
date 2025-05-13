const _SIZE_ GB = 1e9;
const _SIZE_ GiB = 1024 * 1024 * 1024;

double compute_vector_size_GiB(const _SIZE_ size){
    double result = double(size)*sizeof(_TYPE_);
    return result / GiB ;
}

double compute_BW(double duration_ms, _SIZE_ size, size_t N_Repeat){

    double size_tot_GiB = compute_vector_size_GiB(size);    
    return (2*size_tot_GiB * N_Repeat) / (duration_ms / 1000);
    
}