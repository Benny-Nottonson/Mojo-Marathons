from src import Matrix, test_matmul, bench_matmul
from algorithm.functional import vectorize, parallelize

from math import ceil
from sys.info import is_apple_silicon, num_performance_cores, num_physical_cores
from sys.intrinsics import masked_load, masked_store, prefetch, PrefetchOptions
from memory import memset
from sys.info import alignof

# I have 16 ymm registers supposedly (ryzen has 16)
# R means registers and c means cache
# cMatrix = mR * nR = (mR * k) * (k * nR)

# l3 cache 2 total 32mib, 4,193,750 float64 elements, 3 cores have 2_096_875
# l2 cache 6 total 3mib, 393,250 float64 elements, 1 core has ≈ 65541.66667
# l1 cache 6 total 192kib, 24,576 float64 elements, 1 core has 4096

# row-major

# using num_physical_cores or num_performance_cores() gives an error when compiling.
alias NTHREADS = 64


fn matmul_simple[
    Type: DType, M: Int, N: Int, K: Int, //
](inout res: Matrix[Type, M, N], a: Matrix[Type, M, K], b: Matrix[Type, K, N]):
    for m in range(M):
        for k in range(K):
            for n in range(N):
                res[m, n] += a[m, k] * b[k, n]


# --------------


@always_inline
fn pack_panel_b[
    Type: DType,
    K: Int,
    N: Int,
    KC: Int,
    NC: Int, //,
    NR: Int,
](
    b: Matrix[Type, K, N],
    inout block_b: Matrix[Type, KC, NC],
    nr: Int,
    kc: Int,
    j: Int,
    jn: Int,
    pk: Int,
):
    for p in range(kc):

        @parameter
        fn v_iter[nelts: Int](i: Int):
            block_b.data.store[width=nelts](
                j * kc + p * NR + i,
                b.data.load[width=nelts]((pk + p) * N + jn + j + i),
            )

        # for i in range(nr):
        #     block_b.data[j * kc + p * NR + i] = b.data[
        #         (pk + p) * N + jn + j + i
        #     ]
        if nr < NR:
            vectorize[v_iter, 4](nr)
        else:
            vectorize[v_iter, NR](nr)

        @parameter
        fn v_iter_2[nelts: Int](i_temp: Int):
            var i = i_temp + nr
            block_b.data.store[width=nelts](
                j * kc + p * NR + i,
                0,
            )

        vectorize[v_iter_2, 2](NR - nr)
        # for i in range(nr, NR):
        #     block_b.data[j * kc + p * NR + i] = 0


@always_inline
fn pack_block_b[
    Type: DType,
    K: Int,
    N: Int,
    KC: Int,
    NC: Int, //,
    NR: Int,
](
    b: Matrix[Type, K, N],
    inout block_b: Matrix[Type, KC, NC],
    kc: Int,
    nc: Int,
    jn: Int,
    pk: Int,
):
    @parameter
    fn p_iter(jmc: Int):
        var j = jmc * NR
        var nr = min(NR, nc - j)
        pack_panel_b[NR](b, block_b, nr, kc, j, jn, pk)

    var iter = int(ceil(nc / NR))
    parallelize[p_iter](iter, NTHREADS)


@always_inline
fn pack_panel_a[
    Type: DType,
    M: Int,
    K: Int,
    MC: Int,
    KC: Int, //,
    MR: Int,
](
    a: Matrix[Type, M, K],
    inout block_a: Matrix[Type, KC, MC],
    mr: Int,
    kc: Int,
    i: Int,
    pk: Int,
    im: Int,
):
    for p in range(kc):
        for j in range(mr):
            block_a.data[i * kc + p * MR + j] = a.data[
                (im + i + j) * K + pk + p
            ]

        @parameter
        fn v_iter_2[nelts: Int](j_temp: Int):
            var j = j_temp + mr
            block_a.data.store[width=nelts](
                i * kc + p * MR + j,
                0,
            )

        # for j in range(mr, MR):
        #     block_a.data[i * kc + p * MR + j] = 0
        vectorize[v_iter_2, 2](MR - mr)


@always_inline
fn pack_block_a[
    Type: DType,
    M: Int,
    K: Int,
    MC: Int,
    KC: Int, //,
    MR: Int,
](
    a: Matrix[Type, M, K],
    inout block_a: Matrix[Type, KC, MC],
    mc: Int,
    kc: Int,
    pk: Int,
    im: Int,
):
    @parameter
    fn p_iter(imc: Int):
        var i = imc * MR
        var mr = min(MR, mc - i)
        pack_panel_a[MR](a, block_a, mr, kc, i, pk, im)

    var iter = int(ceil(mc / MR))
    parallelize[p_iter](iter, NTHREADS)


fn matmul[
    Type: DType, M: Int, N: Int, K: Int, //
](inout res: Matrix[Type, M, N], a: Matrix[Type, M, K], b: Matrix[Type, K, N]):
    # This was implemented based on https://salykova.github.io/matmul-cpu, so this is not novel, this just implements the idea of doing micro kernels, and using the cache with pack_blocks.
    alias NELTS = simdwidthof[Type]()

    # Normally a cpu has 16 register ymm (ymm = simd), so for the micro kernel we want all the operations to take in the registers. I tried other micro kernel sizes but at least this was the best (and because I don't understand 100% the reason for the shape of micro kernel i don't know what other shape would be better in this case, And i tried lookin in open blas for other sizes, maybe sometimes it is better to use a little bit more registers for the b_matrix?).
    alias MR = 12
    alias NR_MULTIPLIER = 2
    alias NR = NR_MULTIPLIER * NELTS  # float 32 = 2 * 8 = 16

    @parameter
    fn get_NELTS() -> Int:
        var mul = 2
        if is_apple_silicon():
            mul = 4

        var res = simdwidthof[Type]() * mul
        if res > NR:
            return NR
        return res

    alias NELTS_FAST = get_NELTS()

    # for column order
    # kc​×nc block fills the entire L3 cache.
    # mc​×kc​ block fills the entire L2 cache.
    # kc​×nR block fills the entire L1 cache.

    # for row order
    # mc​×kc block fills the entire L3 cache.
    # kc​×nc block fills the entire L2 cache.
    # kc​×mR block fills the entire L1 cache.

    alias NC = NR * 2 # * int((1 / (Type.sizeof() / 8)))
    alias MC = MR * 8  # * int(( 1 / (Type.sizeof() / 8)))
    alias KC = 131_072

    @parameter
    if N * K <= 512 * 512 and M * K <= 512 * 512:
        matmul_2(res, a, b)
        return

    var block_b = Matrix[Type, KC, NC]()
    var block_a = Matrix[Type, KC, MC]()

    alias prefetch_l3_cache = PrefetchOptions().low_locality().to_data_cache()
    alias prefetch_l2_cache = PrefetchOptions().medium_locality().to_data_cache()

    for j in range(0, M, MC):
        var mc = min(MC, M - j)

        for p in range(0, K, KC):
            var kc = min(KC, K - p)
            pack_block_a[MR](a, block_a, mc, kc, p, j)
            prefetch[prefetch_l3_cache](block_a.data)

            for i in range(0, N, NC):
                var nc = min(NC, N - i)
                pack_block_b[NR](b, block_b, kc, nc, i, p)
                prefetch[prefetch_l2_cache](block_b.data)

                @parameter
                fn p_MC_iter(jcr_temp: Int):
                    var jcr = jcr_temp * MR
                    # for jcr in range(0, mc, MR):
                    var mr = min(MR, mc - jcr)

                    for icr in range(0, nc, NR):
                        var nr = min(NR, nc - icr)

                        # using 2 x nelts for the loads and stores in the c_buffer didn't give me better results
                        @parameter
                        @always_inline
                        fn register_kernel():
                            # alignment with booleans gives incorrect values when loading and storing even when specifying the alignment manually
                            var mask = stack_allocation[
                                NR * 2,
                                DType.uint8,
                                alignment = alignof[DType.uint8](),
                            ]()

                            @parameter
                            for h in range(0, NR, NELTS_FAST):
                                mask.store[width=NELTS_FAST](h, 1)

                            @parameter
                            for h in range(NR, 2 * NR, NELTS_FAST):
                                mask.store[width=NELTS_FAST](h, 0)

                            # c uses 12 registers, 2 N and 6 M (there can be different shapes for the kernel)
                            var c_buffer = stack_allocation[NR * MR, Type]()

                            # load values to c_buffer
                            if nr < NR:
                                for jr in range(mr):

                                    @parameter
                                    for ir in range(NR / NELTS):
                                        c_buffer.store(
                                            ir * (MR * NELTS) + jr * NELTS,
                                            masked_load[NELTS](
                                                res.data.offset(
                                                    (j + jcr + jr) * N
                                                    + i
                                                    + icr
                                                    + ir * NELTS
                                                ),
                                                mask.load[width=NELTS](
                                                    NR - nr + ir * NELTS
                                                ).cast[DType.bool](),
                                                SIMD[Type, NELTS](0),
                                                alignment=alignof[Type](),
                                            ),
                                        )
                            else:
                                for jr in range(mr):

                                    @parameter
                                    for ir in range(NR / NELTS):
                                        c_buffer.store(
                                            ir * (MR * NELTS) + jr * NELTS,
                                            res.data.load[width=NELTS](
                                                (j + jcr + jr) * N
                                                + i
                                                + icr
                                                + ir * NELTS
                                            ),
                                        )

                            for pcr in range(kc):

                                @parameter
                                for jr in range(MR):
                                    # 1 register for a
                                    var a_register = block_a.data.load(
                                        jcr * kc + pcr * MR + jr
                                    )

                                    @parameter
                                    for ir in range(NR / NELTS):
                                        # 2 registers for b
                                        var b_register = block_b.data.load[
                                            width=NELTS
                                        ](icr * kc + pcr * NR + ir * NELTS)

                                        c_buffer.store(
                                            ir * (MR * NELTS) + jr * NELTS,
                                            c_buffer.load[width=NELTS](
                                                ir * (MR * NELTS) + jr * NELTS
                                            )
                                            + a_register * b_register,
                                        )

                            # store values from c_buffer
                            if nr < NR:
                                for jr in range(mr):

                                    @parameter
                                    for ir in range(NR / NELTS):
                                        masked_store[NELTS](
                                            c_buffer.load[width=NELTS](
                                                ir * (MR * NELTS) + jr * NELTS
                                            ),
                                            res.data.offset(
                                                (j + jcr + jr) * N
                                                + i
                                                + icr
                                                + ir * NELTS
                                            ),
                                            mask.load[width=NELTS](
                                                NR - nr + ir * NELTS
                                            ).cast[DType.bool](),
                                            alignment=alignof[Type](),
                                        )
                            else:
                                for jr in range(mr):

                                    @parameter
                                    for ir in range(NR / NELTS):
                                        res.data.store(
                                            (j + jcr + jr) * N
                                            + i
                                            + icr
                                            + ir * NELTS,
                                            c_buffer.load[width=NELTS](
                                                ir * (MR * NELTS) + jr * NELTS
                                            ),
                                        )

                        register_kernel()

                var mc_iter = int(ceil(mc / MR))
                parallelize[p_MC_iter](mc_iter, NTHREADS)


fn matmul_2[
    Type: DType, M: Int, N: Int, K: Int, //
](inout res: Matrix[Type, M, N], a: Matrix[Type, M, K], b: Matrix[Type, K, N]):
    alias NELTS = simdwidthof[Type]()

    # Normally a cpu has 16 register ymm (ymm = simd), so for the micro kernel we want all the operations to take in the registers.
    alias MR = 7
    alias NR_MULTIPLIER = 4
    alias NR = NR_MULTIPLIER * NELTS  # float 32 = 2 * 8 = 16

    @parameter
    fn get_NELTS() -> Int:
        var mul = 2
        if is_apple_silicon():
            mul = 4

        var res = simdwidthof[Type]() * mul
        if res > NR:
            return NR
        return res

    alias NELTS_FAST = get_NELTS()

    alias KC = 1000

    # alias prefetch_l3_cache = PrefetchOptions().low_locality().to_data_cache()
    # alias prefetch_l2_cache = PrefetchOptions().medium_locality().to_data_cache()

    # prefetch[prefetch_l3_cache](a.data)
    # prefetch[prefetch_l2_cache](b.data)

    @always_inline
    @parameter
    fn p_M_iter(j_temp: Int):
        var j = j_temp * MR
        # for i in range(0, N, NR):
        var mr = min(MR, M - j)

        for p in range(0, K, KC):
            var kc = min(KC, K - p)

            for i in range(0, N, NR):
                var nr = min(NR, N - i)

                @parameter
                @always_inline
                fn register_kernel():
                    # alignment with booleans gives incorrect values when loading and storing even when specifying the alignment manually
                    var mask = stack_allocation[
                        NR * 2, DType.uint8, alignment = alignof[DType.uint8]()
                    ]()

                    @parameter
                    for h in range(0, NR, NELTS_FAST):
                        mask.store[width=NELTS_FAST](h, 1)

                    @parameter
                    for h in range(NR, 2 * NR, NELTS_FAST):
                        mask.store[width=NELTS_FAST](h, 0)

                    # c uses 12 registers, 2 N and 6 M (there can be different shapes for the kernel)
                    var c_buffer = stack_allocation[NR * MR, Type]()

                    # load values to c_buffer
                    if nr < NR:
                        for jr in range(mr):

                            @parameter
                            for ir in range(NR / NELTS):
                                c_buffer.store(
                                    ir * (MR * NELTS) + jr * NELTS,
                                    masked_load[NELTS](
                                        res.data.offset(
                                            (j + jr) * N + i + ir * NELTS
                                        ),
                                        mask.load[width=NELTS](
                                            NR - nr + ir * NELTS
                                        ).cast[DType.bool](),
                                        SIMD[Type, NELTS](0),
                                        alignment=alignof[Type](),
                                    ),
                                )
                    else:
                        for jr in range(mr):

                            @parameter
                            for ir in range(NR / NELTS):
                                c_buffer.store(
                                    ir * (MR * NELTS) + jr * NELTS,
                                    res.data.load[width=NELTS](
                                        (j + jr) * N + i + ir * NELTS
                                    ),
                                )

                    for pcr in range(kc):
                        for jr in range(mr):
                            # 1 register for a
                            var a_register = a.data.load((j + jr) * K + p + pcr)

                            @parameter
                            for ir in range(NR / NELTS):
                                # 2 registers for b
                                var b_register = b.data.load[width=NELTS](
                                    (p + pcr) * N + i + ir * NELTS
                                )

                                c_buffer.store(
                                    ir * (MR * NELTS) + jr * NELTS,
                                    c_buffer.load[width=NELTS](
                                        ir * (MR * NELTS) + jr * NELTS
                                    )
                                    + a_register * b_register,
                                )

                    # store values from c_buffer
                    if nr < NR:
                        for jr in range(mr):

                            @parameter
                            for ir in range(NR / NELTS):
                                masked_store[NELTS](
                                    c_buffer.load[width=NELTS](
                                        ir * (MR * NELTS) + jr * NELTS
                                    ),
                                    res.data.offset(
                                        (j + jr) * N + i + ir * NELTS
                                    ),
                                    mask.load[width=NELTS](
                                        NR - nr + ir * NELTS
                                    ).cast[DType.bool](),
                                    alignment=alignof[Type](),
                                )
                    else:
                        for jr in range(mr):

                            @parameter
                            for ir in range(NR / NELTS):
                                res.data.store(
                                    (j + jr) * N + i + ir * NELTS,
                                    c_buffer.load[width=NELTS](
                                        ir * (MR * NELTS) + jr * NELTS
                                    ),
                                )

                register_kernel()

    var M_iter = int(ceil(M / MR))

    if M * N * K < 1_000_000:
        for i in range(M_iter):
            p_M_iter(i)
    else:
        parallelize[p_M_iter](M_iter, NTHREADS)


fn main() raises:
    # 550gflops max for my computer (ryzen, 12 threads) with numpy blas
    test_matmul[matmul]()
    bench_matmul[matmul]()
