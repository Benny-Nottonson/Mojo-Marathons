from benchmark import run, keep, clobber_memory
from testing import assert_almost_equal
from algorithm import vectorize
from time import now

alias SCENARIOS = List(
    InlineArray[Int, 3](1, 1, 1),
    InlineArray[Int, 3](1, 47, 97),
    InlineArray[Int, 3](53, 1, 101),
    InlineArray[Int, 3](17, 59, 103),
    InlineArray[Int, 3](1024, 1024, 1024),
    InlineArray[Int, 3](256, 1024, 4096),
    InlineArray[Int, 3](256, 4096, 1024),
    InlineArray[Int, 3](128, 3072, 768),
    InlineArray[Int, 3](1024, 2560, 1024),
    InlineArray[Int, 3](1024, 512, 256),
    InlineArray[Int, 3](1024, 1024, 512),
)

alias TYPES = List(
    DType.int8,
    DType.int16,
    DType.int32,
    DType.int64,
    DType.float16,
    DType.float32,
    DType.float64,
)


fn basic_matmul[
    Type: DType, M: Int, N: Int, K: Int, //
](inout res: Matrix[Type, M, N], a: Matrix[Type, M, K], b: Matrix[Type, K, N]):
    for m in range(M):
        for k in range(K):
            for n in range(N):
                res[m, n] += a[m, k] * b[k, n]


fn test_matmul[MatMul: MatmulSignature]() raises:
    @parameter
    for i in range(len(SCENARIOS)):
        alias SCENARIO = SCENARIOS[i]

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
    for i in range(len(TYPES)):

        @parameter
        for j in range(1, len(SCENARIOS)):
            alias DType = TYPES[i]
            alias Dims = SCENARIOS[j]

            var res = Matrix[DType, Dims[0], Dims[1]]()
            var a = Matrix[DType, Dims[0], Dims[2]].rand()
            var b = Matrix[DType, Dims[2], Dims[1]].rand()

            @parameter
            fn wrap_matmul():
                MatMul(res, a, b)

            clobber_memory()
            var report = run[wrap_matmul]()

            keep(res.data)
            keep(a.data)
            keep(b.data)

            var g_ops = Float64(Dims[0] * Dims[1] * Dims[2] * 2) / 1e9
            var op_type = "I" if DType.is_integral() else "F"

            print(
                "Average G"
                + op_type
                + "op/s:"
                + str(g_ops / report.mean(unit="s")),
                str(DType),
                "dimensions: M="
                + str(Dims[0])
                + ", N="
                + str(Dims[1])
                + ", K="
                + str(Dims[2]),
            )
