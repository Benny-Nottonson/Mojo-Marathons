alias MAX_RANK = 8


@register_passable("trivial")
struct TensorShape:
    var _rank: Int
    var _shape: StaticIntTuple[MAX_RANK]

    @always_inline("nodebug")
    fn __init__(inout self, *shape: Int):
        self._rank = len(shape)
        self._shape = StaticIntTuple[MAX_RANK]()
        for i in range(min(self._rank, MAX_RANK)):
            self._shape[i] = shape[i]

    @always_inline("nodebug")
    fn __init__(inout self, shape: VariadicList[Int]):
        self._rank = len(shape)
        self._shape = StaticIntTuple[MAX_RANK]()
        for i in range(min(self._rank, MAX_RANK)):
            self._shape[i] = shape[i]

    @always_inline("nodebug")
    fn __getitem__(self, index: Int) -> Int:
        return self._shape[index if index >= 0 else self._rank + index]

    @always_inline("nodebug")
    fn __setitem__(inout self, index: Int, value: Int):
        self._shape[index if index >= 0 else self._rank + index] = value

    @always_inline("nodebug")
    fn rank(self) -> Int:
        return self._rank

    @always_inline("nodebug")
    fn num_elements(self) -> Int:
        var result = 1
        for i in range(self._rank):
            result *= self._shape[i]
        return result

    @always_inline("nodebug")
    fn strides(self) -> StaticIntTuple[MAX_RANK]:
        var result = StaticIntTuple[MAX_RANK](0)
        result[self._rank - 1] = 1
        for i in range(self._rank - 2, -1, -1):
            result[i] = result[i + 1] * self._shape[i + 1]
        return result

    @always_inline("nodebug")
    fn __str__(self) -> String:
        var result: String = "("
        for i in range(self._rank):
            result += str(self._shape[i])
            if i < self._rank - 1:
                result += ", "
        result += ")"
        return result

    @always_inline("nodebug")
    fn __eq__(self, other: TensorShape) -> Bool:
        if self.rank() != other.rank():
            return False
        for i in range(self.rank()):
            if self[i] != other[i]:
                return False
        return True

    @always_inline("nodebug")
    fn __ne__(self, other: TensorShape) -> Bool:
        return not self.__eq__(other)


struct Tensor[Type: DType]:
    var _data: DTypePointer[Type]
    var _shape: TensorShape

    @always_inline("nodebug")
    fn __init__(inout self, *dims: Int):
        self._shape = TensorShape(dims)
        self._data = DTypePointer[Type].alloc(self._shape.num_elements())
        memset_zero(self._data, self._shape.num_elements())

    @always_inline("nodebug")
    fn __init__(inout self, owned shape: TensorShape):
        self._data = DTypePointer[Type].alloc(shape.num_elements())
        memset_zero(self._data, shape.num_elements())
        self._shape = shape

    @always_inline("nodebug")
    fn __init__(
        inout self, owned data: DTypePointer[Type], owned shape: TensorShape
    ):
        self._data = data
        self._shape = shape

    @always_inline("nodebug")
    fn __moveinit__(inout self, owned other: Tensor[Type]):
        self._data = other._data
        self._shape = other._shape

    @always_inline("nodebug")
    fn __getitem__(self, index: Int) -> Scalar[Type]:
        return self._data[index]

    @always_inline("nodebug")
    fn __setitem__(self, index: Int, value: Scalar[Type]):
        self._data[index] = value

    @always_inline("nodebug")
    fn data(self) -> DTypePointer[Type]:
        return self._data

    @always_inline("nodebug")
    fn shape(self) -> TensorShape:
        return self._shape

    @always_inline("nodebug")
    fn load[Width: Int](self, index: Int) -> SIMD[Type, Width]:
        return self._data.load[width=Width](index)

    @always_inline("nodebug")
    fn store[Width: Int](self, index: Int, value: SIMD[Type, Width]):
        self._data.store[width=Width](index, value)

    @always_inline("nodebug")
    fn strides(self) -> StaticIntTuple[MAX_RANK]:
        return self._shape.strides()

    @always_inline("nodebug")
    fn rank(self) -> Int:
        return self._shape.rank()

    @always_inline("nodebug")
    fn num_elements(self) -> Int:
        return self._shape.num_elements()

    @always_inline("nodebug")
    fn dim(self, index: Int) -> Int:
        return self._shape[index]

    @always_inline("nodebug")
    fn __del__(owned self):
        self._data.free()
