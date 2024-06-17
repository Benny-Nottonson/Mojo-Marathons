from src import Matrix, Type, Nelts, test_matmul, bench_matmul
from algorithm import vectorize, parallelize

@always_inline("nodebug")
fn matmul[M: Int, N: Int, K: Int, //](inout C: Matrix[Type, M, N], A: Matrix[Type, M, K], B: Matrix[Type, K, N]):
    @parameter
    fn calc_row(m: Int):
        for k in range(A.Cols):
            @parameter
            fn dot[nelts : Int](n : Int):
                C.store[nelts](m,n, C.load[nelts](m,n) + A[m,k] * B.load[nelts](k,n))
            vectorize[dot, Nelts, size = C.Cols]()
    parallelize[calc_row](C.Rows)

fn main() raises:
    test_matmul[matmul]()
    bench_matmul[matmul]()