from benchmark import run, keep, clobber_memory
from testing import assert_almost_equal
from algorithm import vectorize
from time import now

alias SCENARIOS = ((1,1,1), (1,47,97), (53,1,101), (17,59,103), (1024,1024,1024), (256,1024,4096), (256,4096,1024), (128,3072,768), (1024,2560,1024), (1024,512,256), (1024,1024,512))
alias TYPES = (DType.int8, DType.int16, DType.int32, DType.int64, DType.float16, DType.float32, DType.float64)

fn basic_matmul[
    Type: DType, M: Int, N: Int, K: Int, //
](inout res: Matrix[Type, M, N], a: Matrix[Type, M, K], b: Matrix[Type, K, N]):
    for m in range(M):
        for k in range(K):
            var val = a.data[m * K + k]

            fn inner_n[Width: Int](n: Int) capturing:
               res.data.store(n + m * N, b.data.load[width=Width](n + k * N).fma(val, res.data.load[width=Width](n + m * N)))

            vectorize[inner_n, simdwidthof[Type]() * 2, size=N]()

fn test_matmul[matmul: MatmulSignature]() raises:
    @parameter
    for i in range(len(SCENARIOS)):
        alias SCENARIO = SCENARIOS.get[i, Tuple[Int, Int, Int]]()
        alias M = SCENARIO[0]
        alias N = SCENARIO[1]
        alias K = SCENARIO[2]

        var correct = Matrix[Type, M, N]()
        var res = Matrix[Type, M, N]()
        var a = Matrix[Type, M, K].rand()
        var b = Matrix[Type, K, N].rand()

        matmul(res, a, b)
        basic_matmul(correct, a, b)

        for i in range(M * N): 
            assert_almost_equal(res.data[i], correct.data[i], atol=1e-5)

        print("✅ Passed test with M =", M, ", N =", N, ", K =", K)


fn bench_matmul[MatMul: MatmulSignature]() raises:
    print("M, N, K", end=" | ")
    @parameter
    for j in range(1, len(SCENARIOS)):
        alias Dims = SCENARIOS.get[j, Tuple[Int, Int, Int]]()
        print(Dims[0], Dims[1], Dims[2], end=" | ")
    print("Average |\n")

    @parameter
    for i in range(len(TYPES)):
        alias Type = TYPES.get[i, DType]()
        var type_str = str(Type)
        for _ in range(7 - len(type_str)):
            type_str = type_str + " "
        print(type_str, end=" | ")

        var total: Float64 = 0

        @parameter
        for j in range(1, len(SCENARIOS)):
            alias Dims = SCENARIOS.get[j, Tuple[Int, Int, Int]]()
            alias M = Dims[0]
            alias N = Dims[1]
            alias K = Dims[2]

            var res = Matrix[Type, M, N]()
            var a = Matrix[Type, M, K].rand()
            var b = Matrix[Type, K, N].rand()

            fn wrapped_matmul() capturing: 
                MatMul(res, a, b)

            clobber_memory()

            var report = run[wrapped_matmul]()

            keep(res.data)
            keep(a.data)
            keep(b.data)

            var g_ops = Float64(M * N * K * 2) / 1e9
            var length = len(str(M)) + len(str(N)) + len(str(K)) + 2
            var flops = g_ops / report.mean(unit="s")
            total += flops
            print(str(flops)[0:length], end=" | ")
        print(str(total / (len(SCENARIOS) - 1))[0:7], end=" |\n")