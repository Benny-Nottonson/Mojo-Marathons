from random import rand

struct Matrix[Type: DType, Rows: Int, Cols: Int](Movable):
    alias Elements = Rows * Cols
    var data: DTypePointer[Type]

    @staticmethod
    fn rand() -> Self:
        var data = DTypePointer[Type].alloc(Self.Elements)
        rand(data, Self.Elements)
        return Self(data)

    fn __init__(inout self):
        self.data = DTypePointer[Type].alloc(Self.Elements)
        memset_zero(self.data, Self.Elements)

    fn __init__(inout self, data: DTypePointer[Type]):
        self.data = data

    fn __moveinit__(inout self, owned other: Self):
        self.data = other.data

    fn __del__(owned self):
        self.data.free()

    fn __getitem__(self, y: Int, x: Int) -> Scalar[Type]:
        return self.load[1](y, x)

    fn __setitem__(inout self, y: Int, x: Int, value: Scalar[Type]):
        self.store[1](y, x, value)

    fn load[Nelts: Int](self, y: Int, x: Int) -> SIMD[Type, Nelts]:
        return self.data.load[width=Nelts](y * Cols + x)

    fn store[Nelts: Int](inout self, y: Int, x: Int, value: SIMD[Type, Nelts]):
        self.data.store[width=Nelts](y * Cols + x, value)
