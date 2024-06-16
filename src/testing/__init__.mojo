from utils import StaticTuple

from .runner import Runner
from .utils import dot, assert_tensors_equal

alias SCENARIOS = StaticTuple[StaticIntTuple[3], 19](
    # Correctness   
    StaticIntTuple[3](1,1,1),
    StaticIntTuple[3](1,47,97),
    StaticIntTuple[3](53,1,101),
    StaticIntTuple[3](13,53,1),
    StaticIntTuple[3](17,59,103),
    StaticIntTuple[3](23,67,109),
    StaticIntTuple[3](29,71,113),
    StaticIntTuple[3](31,73,127),
    StaticIntTuple[3](37,79,131),
    StaticIntTuple[3](41,83,137),
    StaticIntTuple[3](43,89,139),
    StaticIntTuple[3](417, 1025, 383),

    # Performance
    StaticIntTuple[3](1024, 1024, 1024),
    StaticIntTuple[3](256, 1024, 4096),
    StaticIntTuple[3](256, 4096, 1024),
    StaticIntTuple[3](256, 1024, 1024),
    StaticIntTuple[3](128, 1024, 4096),
    StaticIntTuple[3](128, 4096, 1024),
    StaticIntTuple[3](128, 1024, 1024),
    StaticIntTuple[3](256, 768, 768),
    StaticIntTuple[3](128, 3072, 768),
    StaticIntTuple[3](128, 768, 3072),
    StaticIntTuple[3](256, 3072, 768),
    StaticIntTuple[3](256, 768, 3072),
    StaticIntTuple[3](128, 768, 2304),
    StaticIntTuple[3](1024, 2560, 1024),
    StaticIntTuple[3](1024, 1024, 512),
    StaticIntTuple[3](1024, 352, 512),
    StaticIntTuple[3](1024, 512, 256),
)

@always_inline("nodebug")
fn bench_matmul[matmul: MatmulSignature]():
    Runner[SCENARIOS].test[matmul]()
    Runner[SCENARIOS].gflops[1024, 1024, 1024, matmul](forever=True)
    
    
