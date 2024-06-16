from src import Tensor, TensorShape, Type
from src.testing import bench_matmul

@parameter
@always_inline("nodebug")
fn matmul[
    t1_shape: TensorShape, t2_shape: TensorShape
](inout res: Tensor[Type], t1: Tensor[Type], t2: Tensor[Type]):
    for i in range(t1_shape[0]):
        for j in range(t2_shape[1]):
            for k in range(t1_shape[1]):
                res[i * t2_shape[1] + j] += t1[i * t1_shape[1] + k] * t2[k * t2_shape[1] + j]

fn main():
    bench_matmul[matmul]()