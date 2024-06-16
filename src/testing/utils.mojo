from testing import assert_almost_equal, assert_equal
from algorithm import vectorize, parallelize

@parameter
@always_inline("nodebug")
fn assert_tensors_equal(t1: Tensor[Type], t2: Tensor[Type]) raises:
    assert_equal(t1.num_elements(), t2.num_elements())
    assert_equal(t1.rank(), t2.rank())

    for i in range(t1.rank()):
        assert_equal(t1.dim(i), t2.dim(i))
        
    for i in range(t1.num_elements()):
        assert_almost_equal[Type](t1[i], t2[i], rtol=1e-5)

@parameter
@always_inline("nodebug")
fn dot[
    t1_shape: TensorShape, t2_shape: TensorShape
](inout res: Tensor[Type], t1: Tensor[Type], t2: Tensor[Type]):
    alias M = t1_shape[0]  # t1[0]
    alias K = t1_shape[1]  # t1[1], t2[0]
    alias N = t2_shape[1]  # t2[1]

    alias nelts = simdwidthof[Type]()
    memset_zero[Type](res.data(), res.num_elements())

    @parameter
    fn calc_row(m: Int):
        for k in range(t2_shape[0]):

            @parameter
            fn dot[nelts: Int](n: Int):
                res.store[nelts](
                    m * N + n,
                    res.load[nelts](m * N + n)
                    + t1[m * K + k] * t2.load[nelts](k * N + n),
                )

            vectorize[dot, nelts](N)

    parallelize[calc_row](M, 12)