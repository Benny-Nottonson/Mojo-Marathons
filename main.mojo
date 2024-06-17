from src import Matrix, Type, test_matmul, bench_matmul

fn basic_matmul[M: Int, N: Int, K: Int, //](inout res: Matrix[Type, M, N], a: Matrix[Type, M, K], b: Matrix[Type, K, N]) -> None:
    for y in range(N):
        for x in range(M):
            var sum: Scalar[Type] = 0
            for k in range(K):
                sum += a[x, k] * b[k, y]
            res[x, y] += sum

fn main():
    bench_matmul[basic_matmul, 256]()