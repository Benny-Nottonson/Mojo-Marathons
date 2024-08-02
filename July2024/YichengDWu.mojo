from src import Matrix, test_matmul, bench_matmul
from algorithm.functional import vectorize, parallelize
from sys import has_avx512f, num_performance_cores, is_apple_silicon

alias NUM_THREADS = 6 #  num_performance_cores()
alias L1_ASSOCIATIVITY = 12
alias L1_CACHE_SIZE = 48 * 1024
alias L2_ASSOCIATIVITY = 16
alias L2_CACHE_SIZE = 2 * 1024 * 1024

@always_inline("nodebug")
fn roundup(a: Int, b: Int) -> Int:
    return ((a + b - 1) // b) * b


@always_inline("nodebug")
fn rounddown(a: Int, b: Int) -> Int:
    return (a // b) * b


# math.sqrt doesn't work at compile time
fn intsqrt[n: Int]() -> Int:
    @parameter
    if n == 0:
        return 0
    var x = n
    var y = (x + 1) // 2
    while y < x:
        x = y
        y = (n // x + x) // 2
    return x


@always_inline("nodebug")
fn is_mixed_precision[Type: DType]() -> Bool:
    @parameter
    if Type == DType.bfloat16:
        return True

    @parameter
    if Type == DType.float16:
        return not is_apple_silicon() and not has_avx512f()
    return False

fn matmul_params[Type: DType]() -> StaticIntTuple[5]:
    alias CType = DType.float32 if is_mixed_precision[Type]() else Type
    alias mc = 8192 // sizeof[Type]()  # fix this for simplicity
    alias N = simdwidthof[CType]()
    alias Vectors = 32 if has_avx512f() or is_apple_silicon() else 16

    @parameter
    fn compute_kc[mr: Int, nr: Int]() -> Int:
        alias CBr = int((L1_ASSOCIATIVITY - 1) / (1 + mr / nr))
        return (CBr * L1_CACHE_SIZE) // (nr * sizeof[Type]() * L1_ASSOCIATIVITY)

    @parameter
    fn compute_params[C: Int]() -> StaticIntTuple[5]:
        alias p = C // (intsqrt[C]() + 1)
        alias mr = C // p - 1
        alias nr = p * N
        alias CBr = int((L1_ASSOCIATIVITY - 1) / (1 + mr / nr))
        alias kc = compute_kc[mr, nr]()
        alias nc = (L2_ASSOCIATIVITY - 1) * L2_CACHE_SIZE // (
            kc * sizeof[Type]() * L2_ASSOCIATIVITY
        ) - mr
        return StaticIntTuple[5](mc, nc, kc, mr, nr)

    @parameter
    if Type.is_floating_point():
        alias TempVectors = 1
        return compute_params[Vectors - TempVectors]()
    else:

        @parameter
        if Type == DType.int64:

            @parameter
            if has_avx512f():
                alias TempVectors = 2
                return compute_params[Vectors - TempVectors]()
            else:
                alias TempVectors = 3
                return compute_params[Vectors - TempVectors]()

        elif Type == DType.int8:
            alias TempVectors = 3
            return compute_params[Vectors - TempVectors]()

        else:
            alias TempVectors = 2
            return compute_params[Vectors - TempVectors]()


@always_inline
fn _broadcast[
    Type: DType, //, CType: DType, width: Int
](ptr: DTypePointer[Type]) -> SIMD[CType, width]:
    @parameter
    if is_mixed_precision[Type]():

        @parameter
        if Type == DType.float16:
            return SIMD[CType, width](rebind[Float16](ptr.load()).cast[CType]())
        else:
            constrained[Type == DType.bfloat16, "Incorrect Type"]()
            return SIMD[CType, width](
                rebind[BFloat16](ptr.load()).cast[CType]()
            )
    else:
        constrained[Type == CType, "Type mismatch"]()
        return rebind[SIMD[CType, width]](
            SIMD[Type, simdwidthof[Type]()](ptr.load())
        )


@always_inline
fn _load[
    Type: DType, //, CType: DType, width: Int
](ptr: DTypePointer[Type],) -> SIMD[CType, width]:
    alias alignment = 1 if is_mixed_precision[Type]() else sizeof[
        Type
    ]() * simdwidthof[Type]()

    @parameter
    if is_mixed_precision[Type]():
        var v = ptr.load[
            width = simdwidthof[DType.float32](), alignment=alignment
        ]()
        return rebind[SIMD[CType, width]](v.cast[DType.float32]())
    else:
        constrained[Type == CType, "Type mismatch"]()
        var v = ptr.load[width=width, alignment=alignment]()
        return rebind[SIMD[CType, width]](v)


@value
@register_passable("trivial")
struct Layout:
    var shape: StaticIntTuple[2]
    var strides: StaticIntTuple[2]

    fn __init__(inout self, shape: (Int, Int), strides: (Int, Int)):
        self.shape = StaticIntTuple[2](shape[0], shape[1])
        self.strides = StaticIntTuple[2](strides[0], strides[1])

    fn __init__(inout self, shape: (Int, Int)):
        self.strides = StaticIntTuple[2](shape[1], 1)
        self.shape = StaticIntTuple[2](shape[0], shape[1])

    @always_inline("nodebug")
    fn __call__(self, i: Int, j: Int) -> Int:
        return i * self.strides[0] + j * self.strides[1]

    @always_inline("nodebug")
    fn size(self) -> Int:
        return self.shape[0] * self.shape[1]


struct MatrixView[Type: DType]:
    var data: DTypePointer[Type]
    var layout: Layout

    @always_inline("nodebug")
    fn __init__(inout self, data: DTypePointer[Type], owned layout: Layout):
        self.data = data
        self.layout = layout

    @always_inline("nodebug")
    fn __init__(
        inout self, data: UnsafePointer[Scalar[Type]], shape: (Int, Int)
    ):
        self.data = data
        self.layout = Layout(shape)

    @always_inline("nodebug")
    fn slice(self, i: Int, j: Int, ir: Int, jr: Int) -> Self:
        var shape = (ir, jr)
        var strides = (self.layout.strides[0], self.layout.strides[1])
        var offset = self.layout(i, j)
        return MatrixView(self.data + offset, Layout(shape, strides))

    @always_inline("nodebug")
    fn shape[dim: Int](self) -> Int:
        return self.layout.shape[dim]

    @always_inline("nodebug")
    fn stride[dim: Int](self) -> Int:
        return self.layout.strides[dim]

    fn rand(inout self):
        random.rand(self.data, self.layout.size())

    @always_inline("nodebug")
    fn load[width: Int, *, dim: Int](self, i: Int, j: Int) -> SIMD[Type, width]:
        var offset = self.layout(i, j)
        var ptr = self.data + offset

        @parameter
        if dim == 0:
            return ptr.simd_strided_load[width=width](self.layout.strides[0])
        else:
            return ptr.load[width=width]()

    @always_inline("nodebug")
    fn store[
        width: Int, *, dim: Int
    ](self, value: SIMD[Type, width], i: Int, j: Int):
        var offset = self.layout(i, j)
        var ptr = self.data + offset

        @parameter
        if dim == 0:
            ptr.simd_strided_store[width=width](value, self.layout.strides[0])
        else:
            ptr.store(value)


@always_inline
fn pack_A[
    Type: DType, //, mr: Int
](Ac_buffer: DTypePointer[Type], Ac: MatrixView[Type]) -> MatrixView[Type]:
    @parameter
    fn pack_panel(idx: Int):
        var i = idx * mr
        var dst_ptr = Ac_buffer + i * Ac.shape[1]()
        var src_ptr = Ac.data + i * Ac.stride[0]()

        for p in range(Ac.shape[1]()):

            @parameter
            fn pack_col[width: Int](l: Int):
                (dst_ptr + p * mr).store(
                    l,
                    (src_ptr + p + l * Ac.stride[0]()).simd_strided_load[
                        width=width
                    ](Ac.stride[0]()),
                )

            vectorize[pack_col, simdwidthof[Type]()](min(Ac.shape[0]() - i, mr))
            memset_zero(
                dst_ptr + p * mr + min(Ac.shape[0]() - i, mr),
                mr - min(Ac.shape[0]() - i, mr),
            )

    parallelize[pack_panel]((Ac.shape[0]() + mr - 1) // mr, NUM_THREADS)

    var Ac_layout = Layout(
        (roundup(Ac.shape[0](), mr), Ac.shape[1]()), (1, mr)
    )  # NOTE: The stride is a lie and not used
    return MatrixView(Ac_buffer, Ac_layout)


@always_inline
fn pack_B[
    Type: DType, //, nr: Int
](Bc_buffer: DTypePointer[Type], Bc: MatrixView[Type]) -> MatrixView[Type]:
    alias alignment = 1 if is_mixed_precision[Type]() else sizeof[
        Type
    ]() * simdwidthof[Type]()

    @parameter
    fn pack_panel(idx: Int):
        var i = idx * nr
        var dst_ptr = Bc_buffer + i * Bc.shape[0]()
        var src_ptr = Bc.data + i * Bc.stride[1]()

        for p in range(Bc.shape[0]()):

            @parameter
            fn pack_row[width: Int](l: Int):
                (dst_ptr + p * nr).store[alignment=alignment](
                    l,
                    (src_ptr + p * Bc.stride[0]() + l).load[width=width](),
                )

            vectorize[
                pack_row,
                simdwidthof[Type](),
                unroll_factor = nr // simdwidthof[Type](),
            ](min(Bc.shape[1]() - i, nr))

            memset_zero(
                dst_ptr + p * nr + min(Bc.shape[1]() - i, nr),
                nr - min(Bc.shape[1]() - i, nr),
            )

    parallelize[pack_panel]((Bc.shape[1]() + nr - 1) // nr, NUM_THREADS)

    var Bc_layout = Layout(
        (Bc.shape[0](), roundup(Bc.shape[1](), nr)), (nr, 1)
    )  # NOTE: The stride is a lie and not used
    return MatrixView[Type](Bc_buffer, Bc_layout)


@always_inline
fn matmul_impl[
    Type: DType,
    CType: DType, //,
    mc: Int,
    nc: Int,
    kc: Int,
    mr: Int,
    nr: Int,
](inout C: MatrixView[CType], A: MatrixView[Type], B: MatrixView[Type]):
    var M = C.shape[0]()
    var N = C.shape[1]()
    var K = A.shape[1]()

    var Ac_buffer = DTypePointer[Type].alloc(mc * kc, alignment=64)
    var Bc_buffer = DTypePointer[Type].alloc(kc * nc, alignment=64)

    for i in range(0, A.shape[0](), mc):
        for p in range(0, A.shape[1](), kc):
            var Ab = A.slice(i, p, min(M - i, mc), min(K - p, kc))
            var Ac = pack_A[mr](Ac_buffer, Ab)
            for j in range(0, B.shape[1](), nc):
                var Bb = B.slice(p, j, min(K - p, kc), min(N - j, nc))
                var Bc = pack_B[nr](Bc_buffer, Bb)
                var Cc = C.slice(i, j, min(M - i, mc), min(N - j, nc))

                @parameter
                fn loop_ir(idx: Int):
                    var ir = idx * mr
                    var Ar = MatrixView(
                        Ac.data + ir * Ac.shape[1](), (mr, Ac.shape[1]())
                    )
                    for jr in range(0, Bc.shape[1](), nr):
                        var Cr = Cc.slice(
                            ir,
                            jr,
                            min(Cc.shape[0]() - ir, mr),
                            min(Cc.shape[1]() - jr, nr),
                        )
                        var Br = MatrixView(
                            Bc.data + jr * Bc.shape[0](),
                            (Bc.shape[0](), nr),
                        )
                        if Cr.shape[0]() == mr and Cr.shape[1]() == nr:
                            micro_kernel[mr, nr, False](Cr, Ar, Br)
                        else:
                            micro_kernel[mr, nr, True](Cr, Ar, Br)

                parallelize[loop_ir](
                    (Ac.shape[0]() + mr - 1) // mr, NUM_THREADS
                )

    Bc_buffer.free()
    Ac_buffer.free()


@always_inline
fn micro_kernel[
    Type: DType, CType: DType, //, mr: Int, nr: Int, padding: Bool
](inout Cr: MatrixView[CType], Ar: MatrixView[Type], Br: MatrixView[Type]):
    alias simd_width = simdwidthof[CType]()
    constrained[nr % simd_width == 0, "nr must be multiple of simd_width"]()

    var Ar_ptr = Ar.data
    var Br_ptr = Br.data
    var Cr_ptr = Cr.data

    var ar: SIMD[CType, simd_width]
    var br = InlineArray[SIMD[CType, simd_width], nr // simd_width](
        SIMD[CType, simd_width](0)
    )
    var cr_ptr = stack_allocation[mr * nr, CType, alignment=64]()

    @parameter
    if padding:

        @parameter
        for i in range(mr):
            if i < Cr.shape[0]():

                @parameter
                fn load_col[width: Int](j: Int):
                    DTypePointer.store(
                        cr_ptr + (i * nr + j),
                        (Cr_ptr + (i * Cr.stride[0]() + j)).load[width=width](),
                    )

                vectorize[load_col, simd_width](Cr.shape[1]())
    else:

        @parameter
        for i in range(mr):

            @parameter
            for j in range(0, nr, simd_width):
                cr_ptr.store(
                    i * nr + j,
                    Cr_ptr.load[width=simd_width](i * Cr.stride[0]() + j),
                )

    for p in range(Ar.shape[1]()):

        @parameter
        for j in range(0, nr, simd_width):
            br[j // simd_width] = _load[CType, simd_width](
                Br_ptr + p * nr + j,
            )

        @parameter
        for i in range(mr):

            @parameter
            for j in range(0, nr, simd_width):
                ar = _broadcast[CType, simd_width](Ar_ptr + mr * p + i)
                (cr_ptr + i * nr + j).store(
                    0,
                    ar.fma(
                        br[j // simd_width],
                        (cr_ptr + i * nr + j).load[width=simd_width](),
                    ),
                )

    @parameter
    if padding:

        @parameter
        for i in range(mr):
            if i < Cr.shape[0]():

                @parameter
                fn store_row[width: Int](j: Int):
                    Cr_ptr.store(
                        i * Cr.stride[0]() + j,
                        (cr_ptr + (i * nr + j)).load[width=width](),
                    )

                vectorize[store_row, simd_width](Cr.shape[1]())
    else:

        @parameter
        for i in range(mr):

            @parameter
            for j in range(0, nr, simd_width):
                Cr_ptr.store(
                    i * Cr.stride[0]() + j,
                    (cr_ptr + i * nr + j).load[width=simd_width](),
                )


fn copy[
    M: Int, N: Int, DstType: DType, SrcType: DType, //
](inout dst: Matrix[DstType, M, N], src: Matrix[SrcType, M, N]):
    var elements_per_thread = (M * N + NUM_THREADS - 1) // NUM_THREADS

    @parameter
    fn parallel_copy(idx: Int):
        var start = idx * elements_per_thread
        var end = min((idx + 1) * elements_per_thread, M * N)

        @parameter
        fn vectorized_copy[width: Int](i: Int):
            (dst.data + start + i).store(
                (src.data + start + i).load[width=width]().cast[DstType]()
            )

        vectorize[vectorized_copy, simdwidthof[DstType]()](end - start)

    parallelize[parallel_copy](NUM_THREADS)


fn matmul[
    Type: DType, M: Int, N: Int, K: Int, //
](inout res: Matrix[Type, M, N], a: Matrix[Type, M, K], b: Matrix[Type, K, N]):
    alias params = matmul_params[Type]()
    alias mc = params[0]
    alias nc = params[1]
    alias kc = params[2]
    alias mr = params[3]
    alias nr = params[4]
    alias resized_mc = roundup(min(mc, M), mr)
    alias resized_nc = roundup(min(nc, N), nr)

    var A = MatrixView(a.data, Layout((M, K)))
    var B = MatrixView(b.data, Layout((K, N)))

    @parameter
    if is_mixed_precision[Type]():
        var c = Matrix[Type.float32, M, N]()
        var C = MatrixView(c.data, Layout((M, N)))
        matmul_impl[resized_mc, resized_nc, kc, mr, nr](C, A, B)
        copy(res, c)
    else:
        var C = MatrixView(res.data, Layout((M, N)))
        matmul_impl[resized_mc, resized_nc, kc, mr, nr](C, A, B)


fn main() raises:
    test_matmul[matmul]()
    bench_matmul[matmul]()
