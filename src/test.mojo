from utils import StaticTuple
from random import rand
from testing import assert_almost_equal
from benchmark import keep, clobber_memory, QuickBench, BenchId
from time import now

alias SCENARIOS = StaticTuple[StaticIntTuple[3], 10](
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
)

fn test_matmul[matmul: MatmulSignature]() raises:
    @parameter
    for i in range(len(SCENARIOS)):
        alias M = SCENARIOS[i][0]
        alias N = SCENARIOS[i][1]
        alias K = SCENARIOS[i][2]

        var res = Matrix[Type, M, N]()
        var expected = Matrix[Type, M, N]()
        var a = Matrix[Type, M, K]()
        var b = Matrix[Type, K, N]()

        rand[Type](a.data, a.elements)
        rand[Type](b.data, b.elements)

        clobber_memory()

        matmul(res, a, b)
        Matrix.matmul(expected, a, b)

        for m in range(M):
            for n in range(N):
                assert_almost_equal(res[m, n], expected[m, n])

        print("✅ Passed test with M =", M, ", N =", N, ", K =", K)

fn bench_matmul[matmul: MatmulSignature, Size: Int]() raises:
    var qb = QuickBench()
    qb.run(func=MatmulTester[matmul, Size].run, bench_id=BenchId(func_name="matmul"))
    qb.dump_report()

struct MatmulTester[matmul: MatmulSignature, Size: Int]:
    @staticmethod
    fn run():
        var a = Matrix[Type, Size, Size]()
        var b = Matrix[Type, Size, Size]()
        var res = Matrix[Type, Size, Size]()

        rand[Type](a.data, a.elements)
        rand[Type](b.data, b.elements)

        matmul(res, a, b)