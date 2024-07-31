from src import Matrix, test_matmul, bench_matmul
from algorithm.functional import vectorize, parallelize

from math import ceil
from sys.info import is_apple_silicon, num_performance_cores, num_physical_cores
from sys.intrinsics import masked_load, masked_store
from memory import memset
from sys.info import alignof

# I have 16 ymm registers supposedly (ryzen has 16)
# R means registers and c means cache
# cMatrix = mR * nR = (mR * k) * (k * nR)

# l3 cache 2 total 32mib, 4,193,750 float64 elements
# l2 cache 6 total 3mib, 393,250 float64 elements
# l1 cache 6 total 192kib, 24,576 float64 elements

# row-major

alias NTHREADS = 6


fn matmul[
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


fn matmul_2[
    Type: DType, M: Int, N: Int, K: Int, //
](inout res: Matrix[Type, M, N], a: Matrix[Type, M, K], b: Matrix[Type, K, N]):
    alias NELTS = simdwidthof[Type]()

    # Normally a cpu has 16 register ymm (ymm = simd), so for the micro kernel we want all the operations to take in the registers.
    alias MR = 6
    alias NR = 2 * NELTS  # float 32 = 2 * 8 = 16

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

    alias NC = NR * NTHREADS * 4
    alias MC = MR * NTHREADS * 32  # * int(( 1 / (Type.sizeof() / 8)))
    alias KC = 1000

    @parameter
    if N * K < NC * KC and M * K < MC * KC:
        matmul_5(res, a, b)
        return

    var block_b = Matrix[Type, KC, NC]()
    var block_a = Matrix[Type, KC, MC]()

    for j in range(0, M, MC):
        var mc = min(MC, M - j)

        for p in range(0, K, KC):
            var kc = min(KC, K - p)
            pack_block_a[MR](a, block_a, mc, kc, p, j)

            for i in range(0, N, NC):
                var nc = min(NC, N - i)
                pack_block_b[NR](b, block_b, kc, nc, i, p)

                @parameter
                fn p_MC_iter(jcr_temp: Int):
                    var jcr = jcr_temp * MR
                    # for jcr in range(0, mc, MR):
                    var mr = min(MR, mc - jcr)

                    for icr in range(0, nc, NR):
                        var nr = min(NR, nc - icr)

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


fn matmul_3[
    Type: DType, M: Int, N: Int, K: Int, //
](inout res: Matrix[Type, M, N], a: Matrix[Type, M, K], b: Matrix[Type, K, N]):
    # For now matmul_3 is better than matmul_4 and faster than matmul_2 in some cases
    alias MR = 16

    fn get_NR() -> Int:
        var mul = 2
        if is_apple_silicon():
            mul = 4

        return simdwidthof[Type]() * mul

    alias NR = get_NR()

    @always_inline
    @parameter
    fn p_m(m: Int):
        # for m in range(M):
        for k in range(K):
            var val = a[m, k]

            for n in range(0, N, NR):
                if N - n < NR:

                    @parameter
                    fn nr_iter[nelts: Int](nr: Int):
                        res.data.store(
                            m * N + n + nr,
                            res.data.load[width=nelts](m * N + n + nr)
                            + b.data.load[width=nelts](n + k * N + nr) * val,
                        )

                    vectorize[nr_iter, 4](N - n)
                else:
                    res.data.store(
                        m * N + n,
                        res.data.load[width=NR](m * N + n)
                        + b.data.load[width=NR](k * N + n) * val,
                    )

    if M * N * K < 1_000_000:
        for m in range(M):
            p_m(m)
    else:
        parallelize[p_m](M, NTHREADS)


fn matmul_4[
    Type: DType, M: Int, N: Int, K: Int, //
](inout res: Matrix[Type, M, N], a: Matrix[Type, M, K], b: Matrix[Type, K, N]):
    alias MR = 16

    fn get_NR() -> Int:
        var mul = 2
        if is_apple_silicon():
            mul = 4

        return simdwidthof[Type]() * mul

    alias NR = get_NR()

    # Tiling is an optimization performed for matmul to increase cache locality. The idea is to keep sub-matrices resident in the cache and increase the reuse. The tile function itself can be written in Mojo as:

    # The above will perform 2 dimensional tiling over a 2D iteration space defined to be between ([0,endx],[0,endy])([0,endx​],[0,endy​]). Once we define it above, we can use it within our matmul kernel. For simplicity we choose 4 as the tile height and since we also want to vectorize we use 4 * nelts as the tile width (since we vectorize on the columns).

    alias MC = 256
    alias NC = 128

    for ic in range(0, M, MC):
        var mc = min(MC, M - ic)

        @parameter
        fn p_N_iter(jc_temp: Int):
            var jc = jc_temp * NC
            # for jc in range(0, N, NC):
            var nc = min(NC, N - jc)

            for k in range(K):
                for ir in range(0, mc, MR):
                    var mr = min(MR, mc - ir)

                    for jr in range(0, nc, NR):
                        var nr = min(NR, nc - jr)

                        var c_buffer = stack_allocation[MR * NR, Type]()
                        memset_zero(c_buffer, MR * NR)

                        # for i in range(mr):

                        @parameter
                        fn nr_iter[nelts: Int](j: Int):
                            for i in range(mr):
                                c_buffer.store(
                                    i * NR + j,
                                    c_buffer.load[width=nelts](i * NR + j)
                                    + a.data.load((ic + ir + i) * K + k)
                                    * b.data.load[width=nelts](
                                        k * N + jc + jr + j
                                    ),
                                )

                        if nr == NR:
                            vectorize[nr_iter, NR, size=NR, unroll_factor=1]()
                        else:
                            vectorize[nr_iter, 4](nr)

                        # for i in range(mr):

                        @parameter
                        fn store_iter[nelts: Int](j: Int):
                            for i in range(mr):
                                res.data.store[width=nelts](
                                    (ic + ir + i) * N + jc + jr + j,
                                    res.data.load[width=nelts](
                                        (ic + ir + i) * N + jc + jr + j
                                    )
                                    + c_buffer.load[width=nelts](i * NR + j),
                                )

                        if nr == NR:
                            vectorize[
                                store_iter, NR, size=NR, unroll_factor=1
                            ]()
                        else:
                            vectorize[store_iter, 4](nr)

        var N_iter = int(ceil(N / NC))

        parallelize[p_N_iter](N_iter, NTHREADS)


fn matmul_5[
    Type: DType, M: Int, N: Int, K: Int, //
](inout res: Matrix[Type, M, N], a: Matrix[Type, M, K], b: Matrix[Type, K, N]):
    alias NELTS = simdwidthof[Type]()

    # Normally a cpu has 16 register ymm (ymm = simd), so for the micro kernel we want all the operations to take in the registers.
    alias MR = 6
    alias NR = 2 * NELTS  # float 32 = 2 * 8 = 16

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

    @always_inline
    @parameter
    fn p_N_iter(i_temp: Int):
        var i = i_temp * NR
        # for i in range(0, N, NR):
        var nr = min(NR, N - i)

        for j in range(0, M, MR):
            var mr = min(MR, M - j)

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

                for p in range(K):
                    for jr in range(mr):
                        # 1 register for a
                        var a_register = a.data.load((j + jr) * K + p)

                        @parameter
                        for ir in range(NR / NELTS):
                            # 2 registers for b
                            var b_register = b.data.load[width=NELTS](
                                p * N + i + ir * NELTS
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
                                res.data.offset((j + jr) * N + i + ir * NELTS),
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

    var N_iter = int(ceil(N / NR))

    if M * N * K < 1_000_000:
        for i in range(N_iter):
            p_N_iter(i)
    else:
        parallelize[p_N_iter](N_iter, NTHREADS)


fn main() raises:
    # 550gflops max for my computer (ryzen, 12 threads) with numpy blas
    test_matmul[matmul_2]()
    bench_matmul[matmul_2]()
