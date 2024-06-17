from testing import assert_almost_equal
from time import now

alias SCENARIOS = [
    StaticIntTuple[3](1, 1, 1),                         # Minimal case
    StaticIntTuple[3](1, 47, 97),                       # Small non-square matrices
    StaticIntTuple[3](53, 1, 101),                      # Single row/column
    StaticIntTuple[3](17, 59, 103),                     # Small random sizes
    StaticIntTuple[3](1024, 1024, 1024),                # Large square matrix
    StaticIntTuple[3](256, 1024, 4096),                 # Large rectangular matrix
    StaticIntTuple[3](256, 4096, 1024),                 # Transposed large rectangular matrix
    StaticIntTuple[3](128, 3072, 768),                  # Large non-square matrix
    StaticIntTuple[3](1024, 2560, 1024),                # Large non-square matrix with different dimensions
    StaticIntTuple[3](1024, 512, 256),                  # Large matrix with smaller dimensions
    StaticIntTuple[3](1024, 1024, 512)                  # Large matrix with smaller dimensions
]

fn basic_matmul[Type: DType, M: Int, N: Int, K: Int](inout res: Matrix[Type, M, N], a: Matrix[Type, M, K], b: Matrix[Type, K, N]):
    for m in range(M):
        for k in range(K):
            for n in range(N):
                res[m, n] += a[m, k] * b[k, n]


fn test_matmul[MatMul: MatmulSignature]() raises:
    @parameter
    for i in range(len(SCENARIOS)):
        alias SCENARIO = SCENARIOS.get[i, StaticIntTuple[3]]()
        
        alias M = SCENARIO[0]
        alias N = SCENARIO[1]
        alias K = SCENARIO[2]

        var correct = Matrix[Type, M, N]()
        var res = Matrix[Type, M, N]()
        var a = Matrix[Type, M, K].rand()
        var b = Matrix[Type, K, N].rand()

        MatMul(res, a, b)
        basic_matmul(correct, a, b)

        for m in range(M):
            for n in range(N):
                assert_almost_equal(res[m, n], correct[m, n], atol=1e-5)

        a.data.free()
        b.data.free()
        correct.data.free()
        res.data.free()

        print("✅ Passed test with M =", M, ", N =", N, ", K =", K)

fn bench_matmul[MatMul: MatmulSignature]() raises:
    var res = Matrix[Type, TestSize, TestSize]()
    var a = Matrix[Type, TestSize, TestSize].rand()
    var b = Matrix[Type, TestSize, TestSize].rand()

    alias GFlop = 2 * TestSize * TestSize * TestSize
    var start: Int
    var end: Int

    while True:
        start = now()
        MatMul(res, a, b)
        end = now()

        var elapsed = end - start
        var gflop = GFlop / elapsed

        print("GFlop/s:", gflop)
        memset_zero[Type](res.data, res.elements)