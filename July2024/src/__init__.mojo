from .matrix import Matrix
from .test import test_matmul, bench_matmul

alias Type = DType.float32
alias TestSize = 1024
alias BenchIters = 512
alias MatmulSignature = fn[M: Int, N: Int, K: Int, //](inout Matrix[Type, M, N], Matrix[Type, M, K], Matrix[Type, K, N]) -> None