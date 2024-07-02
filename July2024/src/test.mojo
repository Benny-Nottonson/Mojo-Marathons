from testing import assert_almost_equal
from benchmark import clobber_memory
from algorithm import vectorize
from time import now
import benchmark

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


alias dtypes_to_test = List(DType.int8, DType.int16, DType.int32, DType.int64,DType.float16, DType.float32, DType.float64)


fn basic_matmul[Type: DType, M: Int, N: Int, K: Int, //](inout res: Matrix[Type, M, N], a: Matrix[Type, M, K], b: Matrix[Type, K, N]):    
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
    for i in range(len(dtypes_to_test)):
        @parameter
        for j in range(1, len(SCENARIOS)): # skip the first, not interesting
    
            alias CurrentDType = dtypes_to_test[i]
            alias dimensions = SCENARIOS[j]

            var res = Matrix[CurrentDType, dimensions[0], dimensions[1]]()
            var a = Matrix[CurrentDType, dimensions[0], dimensions[2]].rand()
            var b = Matrix[CurrentDType, dimensions[2], dimensions[1]].rand()

            @parameter
            fn matmul_this():
                # We don't memset to 0 for benchmarking since it has no influence on the performance
                # of the matmul operation.
                MatMul(res, a, b)

            var report = benchmark.run[matmul_this]()

            keep(res)
            keep(a)
            keep(b)
            var g_ops = Float64(dimensions[0] * dimensions[1] * dimensions[2]  * 2) / 1e9

            var op_type: String
            if CurrentDType.is_integral():
                op_type = "I"
            else:
                op_type = "F"
            
            print("Average G" + op_type + "op/s:" + str(g_ops / report.mean(unit="s")), str(CurrentDType),"dimensions: M=" +  str(dimensions[0]) + ", N=" +  str(dimensions[1]) + ", K=" +  str(dimensions[2]))


fn keep(res: Matrix):
    pass