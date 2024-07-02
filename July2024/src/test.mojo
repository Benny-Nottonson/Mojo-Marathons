from testing import assert_almost_equal
from benchmark import clobber_memory
from algorithm import vectorize
from time import now

alias SCENARIOS = [
    StaticIntTuple[3](1, 1, 1),
    StaticIntTuple[3](1, 47, 97),
    StaticIntTuple[3](53, 1, 101),
    StaticIntTuple[3](17, 59, 103),
    StaticIntTuple[3](1024, 1024, 1024),
    StaticIntTuple[3](256, 1024, 4096),
    StaticIntTuple[3](256, 4096, 1024),
    StaticIntTuple[3](128, 3072, 768),
    StaticIntTuple[3](1024, 2560, 1024),
    StaticIntTuple[3](1024, 512, 256),
    StaticIntTuple[3](1024, 1024, 512),
]


alias dtypes_to_test = List[DType](DType.int8, DType.int16, DType.int32, DType.int64,DType.float16, DType.float32, DType.float64)


fn basic_matmul[Type: DType, M: Int, N: Int, K: Int, //](inout res: Matrix[Type, M, N], a: Matrix[Type, M, K], b: Matrix[Type, K, N]):    
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

        for i in range(M * N):
            assert_almost_equal(res.data[i], correct.data[i], atol=1e-5)

        print("✅ Passed test with M =", M, ", N =", N, ", K =", K)


fn bench_matmul[MatMul: MatmulSignature]() raises:

    @parameter
    for i in range(len(dtypes_to_test)):
        alias CurrentDType = dtypes_to_test[i]
        var res = Matrix[CurrentDType, TestSize, TestSize]()
        var a = Matrix[CurrentDType, TestSize, TestSize].rand()
        var b = Matrix[CurrentDType, TestSize, TestSize].rand()

        var start: Int
        var end: Int
        var t: Float64 = 0

        for _ in range(BenchIters):
            clobber_memory()

            start = now()
            MatMul(res, a, b)
            end = now()

            var GFlops = TestSize ** 3 * 2 / (end - start)
            t += GFlops
            print("GFlop/s:", GFlops)

            memset_zero[CurrentDType](res.data, res.Elements)
        
        print("Average GFlop/s:", t / BenchIters, str(CurrentDType))
