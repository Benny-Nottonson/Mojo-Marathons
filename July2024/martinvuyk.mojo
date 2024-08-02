from src import Matrix, test_matmul, bench_matmul
from algorithm.functional import vectorize, parallelize
from algorithm import Static2DTileUnitFunc as Tile2DFunc
from sys import info
from benchmark import clobber_memory
from sys.intrinsics import PrefetchOptions
from math import sqrt


# (i7 7th Gen) passes test, FP16 GFlop/s: 1
fn naive_matmul[
    Type: DType, M: Int, N: Int, K: Int, //
](inout res: Matrix[Type, M, N], a: Matrix[Type, M, K], b: Matrix[Type, K, N]):
    for m in range(M):
        for k in range(K):
            for n in range(N):
                res[m, n] += a[m, k] * b[k, n]


# (i7 7th Gen) doesn't pass test, FP16 GFlop/s: 0.75 (probably because of fma emulation)
fn basic_matmul[
    Type: DType, M: Int, N: Int, K: Int, //
](inout res: Matrix[Type, M, N], a: Matrix[Type, M, K], b: Matrix[Type, K, N]):
    for m in range(M):
        for k in range(K):
            var a_val = a[m, k]

            @parameter
            fn dot[Nelts: Int](n: Int):
                res.store(
                    m, n, b.load[Nelts](k, n).fma(a_val, res.load[Nelts](m, n))
                )

            vectorize[dot, simdwidthof[Type](), size=N]()


# Perform 2D tiling on the iteration space defined by end_x and end_y
fn tile[tiled_fn: Tile2DFunc, tile_x: Int, tile_y: Int](end_x: Int, end_y: Int):
    for y in range(0, end_y, tile_y):
        for x in range(0, end_x, tile_x):
            tiled_fn[tile_x, tile_y](x, y)


# (i7 7th Gen) doesn't pass test, FP16 GFlop/s: 30
fn examples_matmul[
    Type: DType, M: Int, N: Int, K: Int, //
](inout C: Matrix[Type, M, N], A: Matrix[Type, M, K], B: Matrix[Type, K, N]):
    # simdwidth of = amount of `type` elements that fit into a single SIMD register
    # 2x multiplier will use multiple SIMD registers in parallel where possible
    alias nelts = simdwidthof[Type]() * 2
    alias tile_n = 64  # N must be a multiple of this
    alias tile_k = 4  # K must be a multiple of this
    var num_workers: Int = C.Rows

    @parameter
    fn calc_row(m: Int):
        @parameter
        fn calc_tile[tile_x: Int, tile_y: Int](x: Int, y: Int):
            @parameter
            for _k in range(tile_y):
                var k = _k + y

                @parameter
                fn dot[nelts: Int](n: Int):
                    C.store(
                        m,
                        n + x,
                        C.load[nelts](m, n + x)
                        + A[m, k] * B.load[nelts](k, n + x),
                    )

                vectorize[
                    dot, nelts, size=tile_x, unroll_factor = tile_x // nelts
                ]()

        tile[calc_tile, tile_n, tile_k](C.Cols, B.Rows)

    parallelize[calc_row](C.Rows, num_workers)


fn _closest_upper_pow_2(val: Int) -> Int:
    var v = val
    v -= 1
    v |= v >> 1
    v |= v >> 2
    v |= v >> 4
    v |= v >> 8
    v |= v >> 16
    v += 1
    return v


@always_inline("nodebug")
fn tile2[
    tiled_fn: Tile2DFunc, tile_x: Int, tile_y: Int, end_x: Int, end_y: Int
]():
    @parameter
    for y in range(0, end_y, tile_y):
        alias dot_product_iters = tile_y if (
            end_y % tile_y == 0 or tile_y + y < end_y
        ) else end_y - y

        @parameter
        for x in range(0, end_x, tile_x):
            alias vectorize_size = tile_x if (
                end_x % tile_x == 0 or tile_x + x < end_x
            ) else end_x - x
            tiled_fn[vectorize_size, dot_product_iters](x, y)


# (i7 7th Gen) passes test, FP16 GFlop/s: 31-37
fn matmul[
    Type: DType, M: Int, N: Int, K: Int, //
](inout res: Matrix[Type, M, N], a: Matrix[Type, M, K], b: Matrix[Type, K, N]):
    """Generic basis, but number of cores hardcoded for c7g.4xlarge
    (16 vCPU of the 64 cores from Neoverse V1) because
    `info.num_physical_cores()` doesn't work at compile time.

    - Cache:
        - L1 cache: 64 kB (per core)
        - L2 cache: 512 or 1024 kB (per core)
        - L3 cache: 32 or 64 MB (shared)

    Total vector bandwidth: 512 vector bits/cycle (2x256 SVE, 4x128 Neon).

    For `Type.bitwidth() == 64`:
    | specs              | i7 7th Gen   | c7g.4xlarge  | Neoverse V1    |
    |--------------------|--------------|--------------|----------------|
    | num_cores          | 4            | 16           | 64             |
    | nelts              | 8            | 8            | 8              |
    | tile_height        | 4            | 8            | 8              |
    | tile_width         | 256          | 512          | 512            |
    | data dot prod iter | 8 kB         | 32 kB        | 32 kB          |

    For `Type.bitwidth() == 16`:
    | specs              | i7 7th Gen   | c7g.4xlarge  | Neoverse V1    |
    |--------------------|--------------|--------------|----------------|
    | num_cores          | 4            | 16           | 64             |
    | nelts              | 128          | 128          | 128            |
    | tile_height        | 16           | 32           | 32             |
    | tile_width         | 256          | 512          | 512            |
    | data dot prod iter | 8 kB         | 32 kB        | 32 kB          |

    For `Type.bitwidth() == 8`:
    | specs              | i7 7th Gen   | c7g.4xlarge  | Neoverse V1    |
    |--------------------|--------------|--------------|----------------|
    | num_cores          | 4            | 16           | 64             |
    | nelts              | 512          | 512          | 512            |
    | tile_height        | 32           | 64           | 64             |
    | tile_width         | 256          | 512          | 512            |
    | data dot prod iter | 16 kB        | 32 kB        | 32 kB          |

    The calculations are a bit of a guess. I assume most dimensions in a chip
    are of values proportional to each other and a power of 2.
    The main points are:
    - nelts should always be a multiple of `simdwidthof[Type]()` and fit within
        a page (hardcoded to 4 kB for now).
    - data dot prod iter: The data for each dot product calculation (unroll
        factor for vectorization is max(tile_width // nelts, 1))

        `amnt = max(tile_width // nelts, 1) * nelts * tile_height * bitwidth`

        amnt * 2 should be ideally < L1D or L2 or num_cores * amnt < L3 cache. 

    Notes:
        [Information source](\
        https://en.wikichip.org/wiki/arm_holdings/microarchitectures/neoverse_v1
        ).
    """
    # hardcoded, should use generic commented part, but it doesn't compile
    alias num_cores = 16  #  _closest_upper_pow_2(info.num_physical_cores())
    alias simd_width = simdwidthof[Type]()
    # FIXME: need a `info.page_size()` method, 4 kB is pretty common.
    alias page_size_bytes = 4 * 1024
    alias items_per_page = (8 * page_size_bytes) // (
        Type.bitwidth() * info.simdbitwidth()
    )
    alias dtype_bytes = Type.bitwidth() // 8
    alias amnt_matrix_bytes = dtype_bytes * 4 * ((M + K + N) // 3)
    alias nelts = simd_width * (
        items_per_page if amnt_matrix_bytes >= page_size_bytes else 1
    )
    alias tile_height = min(simd_width, _closest_upper_pow_2(K))
    alias tile_width = min(info.simdbitwidth(), _closest_upper_pow_2(N))
    alias num_workers = num_cores

    @parameter
    fn calc_row(m: Int):
        @parameter
        fn calc_tile[tile_x: Int, tile_y: Int](x: Int, y: Int):
            alias factor = max(tile_x // nelts, 1)
            # PrefetchOptions is uncharacteristically underdocumented and has
            # very limited APIs so this is an experiment. I would like to
            # controll how many bytes are fetched into L1D cache and prefetch
            # just the memory region needed for each tile sharing the cache
            # between the different data needed for the calculations (res, a, b)

            # The information I've found: "Fetches the line of data from memory
            # that contains the byte specified with the source operand to a
            # location in the cache hierarchy" and other examples lead me to
            # believe that it fetches 1 "memory-word" (L3 cache bandwidth ?
            # DRAM bitwidth ? the selected cache line size ? OS page size ?)
            # containing the given byte address storing it into the cache.
            # And that it copies the info to lower cache levels as well, so the
            # 3 fetches in a row I do here occupy at least an L2 and 2x L1 cache
            # lines (how big is a cache line ?) and might bring more data into
            # L3, which is useful.
            alias options = (
                PrefetchOptions().to_data_cache().for_read().high_locality()
            )
            res.data.offset(m * N + x).prefetch[options.for_write()]()
            a.data.offset(m * K + y).prefetch[options.medium_locality()]()
            b.data.offset(y * N + x).prefetch[options.medium_locality()]()

            @parameter
            for _k in range(tile_y):
                var k = _k + y

                @parameter
                fn dot[nelts: Int](n: Int):
                    var b_vec = b.load[nelts](k, n + x)
                    var a_sc = a[m, k]
                    var res_acc = res.load[nelts](m, n + x)
                    var value: SIMD[Type, nelts]

                    # CPUs with high width ISA exts like AVX512 or SVE are
                    # most likely to have hardware FP FMA and not emulate it
                    @parameter
                    if Type.is_integral() or (
                        Type.is_floating_point() and info.simdbitwidth() >= 512
                    ):
                        value = b_vec.fma(a_sc, res_acc)
                    else:
                        value = b_vec * a_sc + res_acc

                    res.store(m, n + x, value)

                vectorize[dot, nelts, size=tile_x, unroll_factor=factor]()
                clobber_memory()

        tile2[calc_tile, tile_width, tile_height, N, K]()

    parallelize[calc_row](M, num_workers)


fn main() raises:
    # test_matmul[naive_matmul]()
    # bench_matmul[naive_matmul]()
    # test_matmul[basic_matmul]()
    # bench_matmul[basic_matmul]()
    # test_matmul[examples_matmul]()
    # bench_matmul[examples_matmul]()
    test_matmul[matmul]()
    bench_matmul[matmul]()
