
//Generate an index between [i-max_dist/2 ; i+max_dist/2]
template <typename GeneratorType>
KOKKOS_INLINE_FUNCTION _SIZE_ compute_rindex(const _SIZE_ max_dist,
                                             const _SIZE_ size, const _SIZE_ i,
                                             GeneratorType& generator) {
    _SIZE_ rindex = (i - (max_dist / 2) + generator.rand(max_dist));

    if (rindex > size) {
        rindex = rindex % size;
    }
    if (rindex < 0) {
        rindex = size + rindex;
    }
    return rindex;
}