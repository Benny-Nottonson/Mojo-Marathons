from .tensor import Tensor, TensorShape

alias Type = DType.float32
alias MatmulSignature = fn[t1_shape: TensorShape, t2_shape: TensorShape] (inout Tensor[Type], Tensor[Type], Tensor[Type]) capturing -> None