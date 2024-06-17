from random import rand

@register_passable("trivial")
struct Matrix[Type: DType, Width: Int, Height: Int]:
    alias elements = Width * Height
    var data: DTypePointer[Type]

    @staticmethod
    @always_inline("nodebug")
    fn rand() -> Self:
        var data = DTypePointer[Type].alloc(Self.elements)
        rand(data, Self.elements)
        return Self(data)

    @always_inline("nodebug")
    fn __init__(inout self):
        self.data = DTypePointer[Type].alloc(Self.elements)
        memset_zero(self.data, Self.elements)

    @always_inline("nodebug")
    fn __init__(inout self, data: DTypePointer[Type]):
        self.data = data

    @always_inline("nodebug")
    fn __getitem__(self, x: Int, y: Int) -> Scalar[Type]:
        return self.load[1](x, y)

    @always_inline("nodebug")
    fn __setitem__(inout self, x: Int, y: Int, value: Scalar[Type]):
        self.store[1](x, y, value)

    @always_inline("nodebug")
    fn load[Nelts: Int](self, x: Int, y: Int) -> SIMD[Type, Nelts]:
        return self.data.load[width=Nelts](y * Width + x)

    @always_inline("nodebug")
    fn store[Nelts: Int](inout self, x: Int, y: Int, value: SIMD[Type, Nelts]):
        self.data.store[width=Nelts](y * Width + x, value)