from random import rand


struct Matrix[Type: DType, Rows: Int, Cols: Int]:
    alias Elements = Rows * Cols
    var data: DTypePointer[Type]

    @staticmethod
    @always_inline("nodebug")
    fn rand() -> Self:
        var data = DTypePointer[Type].alloc(Self.Elements)
        rand(data, Self.Elements)
        return Self(data)

    @always_inline("nodebug")
    fn __init__(inout self):
        self.data = DTypePointer[Type].alloc(Self.Elements)
        memset_zero(self.data, Self.Elements)

    @always_inline("nodebug")
    fn __init__(inout self, data: DTypePointer[Type]):
        self.data = data

    @always_inline("nodebug")
    fn __del__(owned self):
        self.data.free()

    @always_inline("nodebug")
    fn __getitem__(self, y: Int, x: Int) -> Scalar[Type]:
        return self.load[1](y, x)

    @always_inline("nodebug")
    fn __setitem__(inout self, y: Int, x: Int, value: Scalar[Type]):
        self.store[1](y, x, value)

    @always_inline("nodebug")
    fn load[Nelts: Int](self, y: Int, x: Int) -> SIMD[Type, Nelts]:
        return self.data.load[width=Nelts](y * Cols + x)

    @always_inline("nodebug")
    fn store[Nelts: Int](inout self, y: Int, x: Int, value: SIMD[Type, Nelts]):
        self.data.store[width=Nelts](y * Cols + x, value)
