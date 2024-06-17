struct Matrix[Type: DType, Width: Int, Height: Int]:
    alias elements = Width * Height
    var data: DTypePointer[Type]

    @always_inline("nodebug")
    fn __init__(inout self):
        self.data = DTypePointer[Type].alloc(Self.elements)
        memset_zero[Type](self.data, Self.elements)

    @always_inline("nodebug")
    fn __init__(inout self, owned data: DTypePointer[Type]):
        self.data = data

    @always_inline("nodebug")
    fn __moveinit__(inout self, owned other: Matrix[Type, Width, Height]):
        self.data = other.data

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
    fn __str__(self) -> String:
        var result: String = "("
        for y in range(Height):
            for x in range(Width):
                result += str(self[x, y])
                if x < Width - 1:
                    result += ", "
            if y < Height - 1:
                result += "\n"
        result += ")"
        return result

    @always_inline("nodebug")
    fn simd_load[Width: Int](self, x: Int, y: Int) -> SIMD[Type, Width]:
        return self.data.load[width=Width](y * Width + x)

    @always_inline("nodebug")
    fn simd_store[Width: Int](inout self, x: Int, y: Int, value: SIMD[Type, Width]):
        self.data.store[width=Width](y * Width + x, value)

    @staticmethod
    @always_inline("nodebug")
    fn matmul[K: Int, //](inout res: Matrix[Type, Width, Height], a: Matrix[Type, Width, K], b: Matrix[Type, K, Height]):
        for m in range(Height):
            for k in range(K):
                for n in range(Width):
                    res[m, n] += a[m, k] * b[k, n]