from src import Matrix, test_matmul, bench_matmul
from algorithm.functional import vectorize, parallelize
from algorithm import parallel_memcpy

# mojo --version
# mojo 24.4.0 (2cb57382)


fn matmul[
    Type: DType, M: Int, N: Int, K: Int, //
](inout res: Matrix[Type, M, N], a: Matrix[Type, M, K], b: Matrix[Type, K, N]):
    alias Width: Int = simdwidthof[Type]()
    alias Elements: Int = 32 * 1024 // sizeof[Type]()
    alias BLOCK_N: Int = Elements // Width * Width
    alias BLOCK_M: Int = Elements // BLOCK_N

    # Calculate a block of the result matrix
    fn calculate_block[BM: Int, BN: Int](bm: Int, bn: Int) capturing:
        var acc = stack_allocation[BM * BN, Type]()
        memset_zero(acc, BM * BN)

        for k in range(K):
            var b = b.data + k * N

            for m in range(BM):
                var a_val = a[bm + m, k]
                var acc = acc + m * BN

                fn inner_n[W: Int](n: Int) capturing:
                    acc.store[width=W](
                        n,
                        b.load[width=W](bn + n).fma(
                            a_val, acc.load[width=W](n)
                        ),
                    )

                vectorize[inner_n, Width * 2, size=BN]()

        for m in range(BM):
            parallel_memcpy(res.data + (bm + m) * N + bn, acc + m * BN, BN)

    # Process a block of the matrix
    fn process_block[BM: Int](bm: Int) capturing:
        @parameter
        for bn in range(0, N if BM else 0, BLOCK_N):
            calculate_block[BM, min(BLOCK_N, N - bn)](
                bm * BM if BM == BLOCK_M else bm, bn
            )

    # Parallelize the processing of blocks
    parallelize[process_block[BLOCK_M]](M // BLOCK_M)
    process_block[M % BLOCK_M](M - M % BLOCK_M)

    # Parallelize the processing of remaining blocks
    parallelize[process_block[BLOCK_M]](M // BLOCK_M)
    process_block[M % BLOCK_M](M - M % BLOCK_M)


fn main() raises:
    test_matmul[matmul]()
    bench_matmul[matmul]()
