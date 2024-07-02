from .matrix import Matrix
from .test import test_matmul, bench_matmul

alias Type = DType.float16
alias TestSize = 1024
alias BenchIters = 16
alias MatmulSignature = fn[Type: DType, M: Int, N: Int, K: Int, //](inout Matrix[Type, M, N], Matrix[Type, M, K], Matrix[Type, K, N]) -> None