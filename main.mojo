from src import Matrix, Type, Nelts, test_matmul, bench_matmul
from algorithm import vectorize, parallelize


@always_inline("nodebug")
fn matmul[M: Int, N: Int, K: Int, //](inout res: Matrix[Type, M, N], a: Matrix[Type, M, K], b: Matrix[Type, K, N]):
    @parameter
    fn calc_row(m: Int):
        for k in range(K):
            var a_val = a[m, k]

            @parameter
            fn dot[nelts: Int](n: Int):
                res.store[nelts](m, n, res.load[nelts](m, n) + a_val * b.load[nelts](k, n))

            vectorize[dot, Nelts, size=N]()

    parallelize[calc_row](M)


fn main() raises:
    test_matmul[matmul]()
    bench_matmul[matmul]()
