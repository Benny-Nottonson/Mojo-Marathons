from benchmark import run, keep, clobber_memory, Report
from testing import assert_almost_equal
from algorithm import vectorize
from time import now
from sys.info import is_x86, has_sse4, has_avx, has_avx2, has_avx512f, has_vnni, is_apple_silicon, is_apple_m1, is_apple_m2, is_apple_m3, has_neon, has_neon_int8_dotprod, has_neon_int8_matmul, num_physical_cores, num_logical_cores, num_performance_cores, simdbitwidth, os_is_macos, os_is_linux, os_is_windows, is_little_endian, is_64bit

alias SCENARIOS = InlineArray[size=10](
    InlineArray[size=3](1, 1, 1), 
    InlineArray[size=3](1, 47, 97), 
    InlineArray[size=3](53, 1, 101), 
    InlineArray[size=3](17, 59, 103), 
    InlineArray[size=3](128, 128, 128), 
    InlineArray[size=3](128, 3072, 768), 
    InlineArray[size=3](512, 512, 512), 
    InlineArray[size=3](256, 1024, 4096), 
    InlineArray[size=3](1024, 1024, 1024), 
    InlineArray[size=3](4096, 4096, 8192)
)
alias TYPES = InlineArray[size=7](DType.int8, DType.int16, DType.int32, DType.int64, DType.float16, DType.float32, DType.float64)

fn print_system_specs():
    print("System Specs", end=" | ")
    print("CPU: ", end="")
    if is_x86():
        print("x86", end=" ")
        if has_sse4(): print("SSE4", end=" ")
        if has_avx(): print("AVX", end=" ")
        if has_avx2(): print("AVX2", end=" ")
        if has_avx512f(): print("AVX512", end=" ")
        if has_vnni(): print("VNNI", end=" ")
        print("", end="| ")
    elif is_apple_silicon():
        print("Apple Silicon", end=" ")
        if is_apple_m1(): print("M1", end=" ")
        elif is_apple_m2(): print("M2", end=" ")
        elif is_apple_m3(): print("M3", end=" ")
        print("", end="| ")
    elif has_neon():
        print("ARM Neon", end=" ")
        if has_neon_int8_dotprod(): print("DotProd", end=" ")
        if has_neon_int8_matmul(): print("I8MM", end=" ")
        print("", end=" | ")
    print("Cores: Physical =", num_physical_cores(), "- Logical =", num_logical_cores(), "- Performance =", num_performance_cores(), end=" | ")
    print("SIMD width:", simdbitwidth(), "bits", end=" | ")
    print("OS: ", end=" ")
    if os_is_macos(): print("macOS", end=" | ")
    elif os_is_linux(): print("Linux", end=" | ")
    elif os_is_windows(): print("Windows", end=" | ")
    else: print("Unknown", end=" | ")
    print("Endianness:", "Little" if is_little_endian() else "Big", end=" | ")
    print("Bit width:", "64-bit" if is_64bit() else "32-bit")

fn basic_matmul[Type: DType, M: Int, N: Int, K: Int](inout res: Matrix[Type, M, N], a: Matrix[Type, M, K], b: Matrix[Type, K, N]):
    for m in range(M):
        for k in range(K):
            var val = a.data[m * K + k]
            fn inner_n[Width: Int](n: Int) capturing:
               res.data.store(n + m * N, b.data.load[width=Width](n + k * N).fma(val, res.data.load[width=Width](n + m * N)))
            vectorize[inner_n, simdwidthof[Type]() << (1 + is_apple_silicon()), size=N]()

fn test_matmul[matmul: MatmulSignature]() raises:
    @parameter
    for i in range(len(SCENARIOS) // 2):
        alias SCENARIO = SCENARIOS[i]
        var correct = Matrix[Type, SCENARIO[0], SCENARIO[1]]()
        var res = Matrix[Type, SCENARIO[0], SCENARIO[1]]()
        var a = Matrix[Type, SCENARIO[0], SCENARIO[2]].rand()
        var b = Matrix[Type, SCENARIO[2], SCENARIO[1]].rand()
        matmul(res, a, b)
        basic_matmul(correct, a, b)
        for i in range(SCENARIO[0] * SCENARIO[1]): 
            assert_almost_equal(res.data[i], correct.data[i], atol=1e-5)
        print("✅ Passed test with M =", SCENARIO[0], ", N =", SCENARIO[1], ", K =", SCENARIO[2])

fn bench_matmul[MatMul: MatmulSignature]() raises:
    print_system_specs()

    print("M, N, K", end=" | ")
    for j in range(1, len(SCENARIOS)):
        print(SCENARIOS[j][0], SCENARIOS[j][1], SCENARIOS[j][2], end=" | ")
    print("Average |")

    @parameter
    for i in range(len(TYPES)):
        alias Type = TYPES[i]
        var total: Float64 = 0
        print(str(Type), end="")
        for _ in range(7 - len(str(Type))): print(" ", end="")
        print(" | ", end="")
        @parameter
        for j in range(1, len(SCENARIOS)):
            alias Dims = SCENARIOS[j]
            var res = Matrix[Type, Dims[0], Dims[1]]()
            var a = Matrix[Type, Dims[0], Dims[2]].rand()
            var b = Matrix[Type, Dims[2], Dims[1]].rand()
            fn wrapped_matmul() capturing: MatMul(res, a, b)
            clobber_memory()
            var report = run[wrapped_matmul]()
            var flops = Float64(Dims[0] * Dims[1] * Dims[2] * 2) / 1e9 / report.mean(unit="s")
            keep(res.data)
            keep(a.data)
            keep(b.data)
            total += flops
            print(str(flops)[0:7], end="")
            for _ in range(len(str(Dims[0])) + len(str(Dims[1])) + len(str(Dims[2])) + 2 - len(str(flops)[0:7])):
                print(" ", end="")
            print(" | ", end="")
        print(str(total / (len(SCENARIOS) | 1))[0:7], end=" |\n")
