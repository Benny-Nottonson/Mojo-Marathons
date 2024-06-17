from .matrix import Matrix
from .test import test_matmul, bench_matmul

alias Type = DType.float32
alias Nelts = simdwidthof[Type]()
alias TestSize = 1024
alias MatmulSignature = fn[M: Int, N: Int, K: Int, //](inout Matrix[Type, M, N], Matrix[Type, M, K], Matrix[Type, K, N]) -> None