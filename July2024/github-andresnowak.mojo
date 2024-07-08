from src import Matrix, test_matmul, bench_matmul
from algorithm.functional import vectorize, parallelize

from math import ceil

# I have 16 ymm registers supposedly
# R means registers and c means cache
# cMatrix = mR * nR = (mR * k) * (k * nR)

# l3 cache 2 total 32mib, 4,193,750 float64 elements
# l2 cache 6 total 3mib, 393,250 float64 elements
# l1 cache 6 total 192kib, 24,576 float64 elements

# row-major


fn matmul[
    Type: DType, M: Int, N: Int, K: Int, //
](inout res: Matrix[Type, M, N], a: Matrix[Type, M, K], b: Matrix[Type, K, N]):
    for m in range(M):
        for k in range(K):
            for n in range(N):
                res[m, n] += a[m, k] * b[k, n]


# --------------


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
        for i in range(nr):
            block_b.data[j * kc + p * NR + i] = b.data[
                (pk + p) * N + jn + j + i
            ]
        for i in range(nr, NR):
            block_b.data[j * kc + p * NR + i] = 0


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
    parallelize[p_iter](iter)


fn pack_panel_a[
    Type: DType,
    M: Int,
    K: Int,
    MC: Int,
    KC: Int, //,
    MR: Int,
](
    a: Matrix[Type, M, K],
    inout block_a: Matrix[Type, MC, KC],
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
        for j in range(mr, MR):
            block_a.data[i * kc + p * MR + j] = 0


fn pack_block_a[
    Type: DType,
    M: Int,
    K: Int,
    MC: Int,
    KC: Int, //,
    MR: Int,
](
    a: Matrix[Type, M, K],
    inout block_a: Matrix[Type, MC, KC],
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
    parallelize[p_iter](iter)


fn matmul_2[
    Type: DType, M: Int, N: Int, K: Int, //
](inout res: Matrix[Type, M, N], a: Matrix[Type, M, K], b: Matrix[Type, K, N]):
    alias NTHREADS = 6

    alias MR = 16
    alias NR = simdwidthof[Type]() * 2

    alias NC = MR * NTHREADS * 4
    alias MC = NR * NTHREADS * 32
    alias KC = 1000

    var block_b = Matrix[Type, KC, NC]()
    var block_a = Matrix[Type, MC, KC]()

    for j in range(0, N, NC):
        var nc = min(NC, N - j)

        for p in range(0, K, KC):
            var kc = min(KC, K - p)
            pack_block_b[NR](b, block_b, kc, nc, j, p)

            for i in range(0, M, MC):
                var mc = min(MC, M - i)
                pack_block_a[MR](a, block_a, mc, kc, p, i)

                @parameter
                fn p_iter(jcp: Int):
                    var jc = jcp * NR
                    # for jc in range(0, nc, NR):
                    for ic in range(0, mc, MR):
                        var nr = min(NR, nc - jc)
                        var mr = min(MR, mc - ic)

                        var c_buffer = stack_allocation[MR * NR, Type]()

                        # do the computation
                        for kr in range(kc):

                            @parameter
                            fn v_iter[nelts: Int](jr: Int):
                                # for jr in range(nr):
                                for ir in range(mr):
                                    res.data.store[width=nelts](
                                        (i + ic + ir) * N + j + jc + jr,
                                        res.data.load[width=nelts](
                                            (i + ic + ir) * N + j + jc + jr
                                        )
                                        + block_a.data[ic * kc + kr * MR + ir]
                                        * block_b.data.load[width=nelts](
                                            jc * kc + kr * NR + jr
                                        ),
                                    )

                            vectorize[v_iter, NR](nr)

                var iter = int(ceil(nc / NR))
                parallelize[p_iter](iter)


fn main() raises:
    # 500gflops max for my computer
    test_matmul[matmul_2]()
    bench_matmul[matmul_2]()
