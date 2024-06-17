@register_passable
struct Matrix[Type: DType, Width: Int, Height: Int]:
    alias elements = Width * Height
    var data: DTypePointer[Type]

    @always_inline("nodebug")
    fn __init__(inout self):
        self.data = DTypePointer[Type].alloc(Self.elements)

    @always_inline("nodebug")
    fn __getitem__(self, x: Int, y: Int) -> Scalar[Type]:
        return self.data[y * Width + x]

    @always_inline("nodebug")
    fn __setitem__(inout self, x: Int, y: Int, value: Scalar[Type]):
        self.data[y * Width + x] = value

    @always_inline("nodebug")
    fn __del__(owned self):
        self.data.free()

    @always_inline("nodebug")
    fn simd_load[Width: Int](self, x: Int, y: Int) -> SIMD[Type, Width]:
        return self.data.load[width=Width](y * Width + x)

    @always_inline("nodebug")
    fn simd_store[Width: Int](inout self, x: Int, y: Int, value: SIMD[Type, Width]):
        self.data.store[width=Width](y * Width + x, value)