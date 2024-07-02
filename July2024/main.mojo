from src import Matrix, Type, test_matmul, bench_matmul
from algorithm.functional import vectorize, parallelize

fn matmul[Type: DType, M: Int, N: Int, K: Int, //](inout res: Matrix[Type, M, N], a: Matrix[Type, M, K], b: Matrix[Type, K, N]):    
    for m in range(M):
        for k in range(K):
            for n in range(N):
                res[m, n] += a[m, k] * b[k, n]

fn main() raises:
    #test_matmul[matmul]()
    bench_matmul[matmul]()
