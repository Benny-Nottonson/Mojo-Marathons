from .matrix import Matrix
from .test import test_matmul, bench_matmul

alias Type = DType.float32
alias Width = simdwidthof[Type]()
alias MatmulSignature = fn[M: Int, N: Int, K: Int, //](inout Matrix[Type, M, N], Matrix[Type, M, K], Matrix[Type, K, N]) -> None