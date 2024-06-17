from utils import StaticTuple
from random import rand
from testing import assert_almost_equal
from benchmark import keep, clobber_memory
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
    StaticIntTuple[3](1024, 512, 256)                   # Large matrix with smaller dimensions
)

fn test_matmul[matmul: MatmulSignature]() raises:
    @parameter
    for i in range(len(SCENARIOS)):
        print("Testing matmul with M =", SCENARIOS[i][0], ", N =", SCENARIOS[i][1], ", K =", SCENARIOS[i][2])

        var start = now()

        var res = Matrix[Type, SCENARIOS[i][0], SCENARIOS[i][1]]()
        var expected = Matrix[Type, SCENARIOS[i][0], SCENARIOS[i][1]]()
        var a = Matrix[Type, SCENARIOS[i][0], SCENARIOS[i][2]]()
        var b = Matrix[Type, SCENARIOS[i][2], SCENARIOS[i][1]]()

        rand[Type](a.data, a.elements)
        rand[Type](b.data, b.elements)
        rand[Type](res.data, res.elements)
        memcpy[res.elements](expected.data, res.data)

        clobber_memory()

        expected.matmul(a, b)
        matmul(res, a, b)

        for y in range(SCENARIOS[i][1]):
            for x in range(SCENARIOS[i][0]):
                assert_almost_equal(res[x, y], expected[x, y], atol=1e-6)

        keep(a.data)
        keep(b.data)
        keep(res.data)
        keep(expected.data)

        var elapsed = (now() - start) / 1e9

        print("✅Test passed in ", elapsed, " seconds")

fn bench_matmul[matmul: MatmulSignature, Size: Int]():
    var a = Matrix[Type, Size, Size]()
    var b = Matrix[Type, Size, Size]()
    var res = Matrix[Type, Size, Size]()

    rand[Type](a.data, a.elements)
    rand[Type](b.data, b.elements)
    rand[Type](res.data, res.elements)

    clobber_memory()
    
    alias Flop = 2 * Size * Size * Size
    var start: Int
    var end: Int
    var total = 0.0
    var i = 0

    while True:
        i += 1
        start = now()
        matmul(res, a, b)
        end = now() - start
        var elapsed = end / 1e9
        var gflops = (Flop / elapsed) / 1e9
        total += gflops
        print("Iteration ", i, ":", gflops, "GFLOPS")
        print("Average: ", total / i, "GFLOPS")
