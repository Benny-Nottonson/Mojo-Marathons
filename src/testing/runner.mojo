from utils import StaticTuple
from random import rand
from benchmark import keep, clobber_memory
from time import now

struct Runner[
    size: Int, //,
    scenarios: StaticTuple[StaticIntTuple[3], size],
]:
    @staticmethod
    fn test[
        matmul: MatmulSignature
    ]():
        @parameter
        fn test_unroll[i: Int]():
            alias M = scenarios[i][0]
            alias N = scenarios[i][1]
            alias K = scenarios[i][2]
            Self.check[M, N, K, matmul]()

        unroll[test_unroll, size]()

    @staticmethod
    fn check[
        M: Int,
        N: Int,
        K: Int,
        matmul: MatmulSignature,
    ]():
        alias t1_shape = TensorShape(M, K)
        alias t2_shape = TensorShape(K, N)
        alias res_shape = TensorShape(M, N)

        var t1 = Tensor[Type](t1_shape)
        var t2 = Tensor[Type](t2_shape)
        var res = Tensor[Type](res_shape)
        var res_true = Tensor[Type](res_shape)
        rand[Type](t1.data(), t1.num_elements())
        rand[Type](t2.data(), t2.num_elements())

        clobber_memory()
        
        matmul[t1_shape, t2_shape](res, t1, t2)
        dot[t1_shape, t2_shape](res_true, t1, t2)

        try:
            assert_tensors_equal(res, res_true)
            print("✅: ", t1_shape, "*", t2_shape, "->", res_shape)
        except e:
            print("❌: ", t1_shape, "*", t2_shape, "->", res_shape)

        keep(res.data())
        keep(t1.data())
        keep(t2.data())

    @staticmethod
    fn gflops[
        M: Int,
        N: Int,
        K: Int,
        matmul: MatmulSignature,
    ](forever: Bool = False):
        Self.check[M, N, K, matmul]()

        alias t1_shape = TensorShape(M, K)
        alias t2_shape = TensorShape(K, N)
        alias res_shape = TensorShape(M, N)

        var t1 = Tensor[Type](t1_shape)
        var t2 = Tensor[Type](t2_shape)
        var res = Tensor[Type](res_shape)

        rand[Type](t1.data(), t1.num_elements())
        rand[Type](t2.data(), t2.num_elements())

        alias flop = 2 * N * M * K
        var st: Int
        var et: Int

        clobber_memory()

        if forever:
            var gflops = 0.0
            for i in range(1_000_000):
                st = now()
                matmul[t1_shape, t2_shape](res, t1, t2)
                et = now()
                var gflop = flop / (et - st)
                gflops += gflop
                print("GFLOPS: ", gflop)
                print("AVERAGE: ", gflops / i)    
        else:
            var gflops = 0.0
            for i in range(500):
                st = now()
                matmul[t1_shape, t2_shape](res, t1, t2)
                et = now()
                var gflop = flop / (et - st)
                gflops += gflop
                print("GFLOPS: ", gflop)
                print("AVERAGE: ", gflops / i)
                
        keep(res.data())
        keep(t1.data())
        keep(t2.data())