from src import Matrix, Type, Nelts, test_matmul, bench_matmul
from algorithm import vectorize, parallelize

fn matmul[M: Int, N: Int, K: Int, //](inout res: Matrix[Type, M, N], a: Matrix[Type, M, K], b: Matrix[Type, K, N]):
    ...


fn main() raises:
    test_matmul[matmul]()
    bench_matmul[matmul]()
