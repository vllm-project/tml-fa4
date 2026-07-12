# Copyright (c) 2026, Colfax International.
#
# Modified from CUTLASS code, original copyright:
# Copyright (c) 2025 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.

# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.

# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

from enum import Enum, member
from typing import Callable, Tuple, Type, Union
import cuda.bindings.driver as cuda

import cutlass
from cutlass import const_expr, Int32, Float32, Constexpr
from cutlass.base_dsl import dsl_user_op
import cutlass.cute as cute
from cutlass.cute import testing
import cutlass.utils as utils
from cutlass.utils.gemm.sm100 import (
    transform_partitioned_tensor_layout,
    epilogue_tmem_copy_and_partition,
    epilogue_smem_copy_and_partition,
)
import cutlass.pipeline as pipeline
from cutlass.pipeline import pipeline_init_arrive, pipeline_init_wait
from cutlass.cute.nvgpu import cpasync, tcgen05
from cutlass._mlir import ir
from cutlass._mlir.dialects import arith, llvm

from quack.activation import silu as _quack_silu, relu as _quack_relu

#######################################
#     Example epilogue functions      #
#######################################


# Factory: turn operations on Float32s to operations on TensorSSAs
def _vectorize(fn: Callable):
    @dsl_user_op
    def inner(vec: cute.TensorSSA, *, loc=None, ip=None):
        length = cute.size(vec.shape)
        f32 = Float32.mlir_type
        vec_dst = llvm.mlir_zero(ir.VectorType.get([length], f32, loc=loc), loc=loc, ip=ip)
        for i in range(length):
            i0 = arith.constant(Int32.mlir_type, i, loc=loc, ip=ip)
            x0 = Float32(llvm.extractelement(vec, i0, loc=loc, ip=ip))
            r = fn(x0)
            vec_dst = llvm.insertelement(vec_dst, r.ir_value(), i0, loc=loc, ip=ip)
        return cute.TensorSSA(vec_dst, vec.shape, Float32)

    return inner


# Same but for pairs of Float32s (lowering to PTX float32x2 instructions)
def _vectorize_packed(fn: Callable):
    @dsl_user_op
    def inner(vec: cute.TensorSSA, *, loc=None, ip=None):
        length = cute.size(vec.shape)
        f32 = Float32.mlir_type
        vec_dst = llvm.mlir_zero(ir.VectorType.get([length], f32, loc=loc), loc=loc, ip=ip)
        for i in range(0, length, 2):
            i0 = arith.constant(Int32.mlir_type, i, loc=loc, ip=ip)
            i1 = arith.constant(Int32.mlir_type, i + 1, loc=loc, ip=ip)
            x0 = Float32(llvm.extractelement(vec, i0, loc=loc, ip=ip))
            x1 = Float32(llvm.extractelement(vec, i1, loc=loc, ip=ip))
            r = fn((x0, x1))
            vec_dst = llvm.insertelement(vec_dst, r[0].ir_value(), i0, loc=loc, ip=ip)
            vec_dst = llvm.insertelement(vec_dst, r[1].ir_value(), i1, loc=loc, ip=ip)
        return cute.TensorSSA(vec_dst, vec.shape, Float32)

    return inner


silu_epilogue = _vectorize_packed(_quack_silu)
sigmoid_epilogue = _vectorize_packed(_quack_silu)
relu_epilogue = _vectorize_packed(_quack_relu)

relu_epilogue_unpacked = _vectorize(_quack_relu)


# Storing functions in enum avoids spurious recompilation
class EpilogueFunction(Enum):
    SILU = member(silu_epilogue)
    SIGMOID = member(sigmoid_epilogue)
    RELU = member(relu_epilogue)
    RELU_UNPACKED = member(relu_epilogue_unpacked)


#######################################
#              Kernel                 #
#######################################


class MixedDtypeGemmKernel:
    def __init__(
        self,
        a_dtype: Type[cutlass.Numeric],
        b_dtype: Type[cutlass.Numeric],
        mma_dtype: Type[cutlass.Numeric],
        acc_dtype: Type[cutlass.Numeric],
        c_dtype: Type[cutlass.Numeric],
        use_2cta_instrs: bool,
        mma_tiler_mn: Tuple[int, int],
        cluster_shape_mn: Tuple[int, int],
    ):
        self.a_dtype = a_dtype
        self.b_dtype = b_dtype
        self.mma_dtype = mma_dtype
        self.acc_dtype = acc_dtype
        self.c_dtype = c_dtype
        self.use_2cta_instrs = use_2cta_instrs
        self.cluster_shape_mn = cluster_shape_mn
        # K dimension is deferred in _setup_attributes
        self.mma_tiler_mn = mma_tiler_mn
        self.mma_tiler = (*mma_tiler_mn, 1)
        self.arch = "sm_100"

        self.cta_group = tcgen05.CtaGroup.TWO if use_2cta_instrs else tcgen05.CtaGroup.ONE

        self.do_transform_a = self.a_dtype != self.mma_dtype and not (
            self.a_dtype == cutlass.Float32 and self.mma_dtype == cutlass.TFloat32
        )
        self.do_transform_b = self.b_dtype != self.mma_dtype and not (
            self.b_dtype == cutlass.Float32 and self.mma_dtype == cutlass.TFloat32
        )

        self.occupancy = 1
        # Set specialized warp ids
        self.epilogue_warp_id = (0, 1, 2, 3)
        self.mma_warp_id = 4
        self.tma_warp_id = 5
        self.sched_warp_id = 6
        self.unused_warp_id = 7
        self.num_convert_warps = 8
        self.convert_warp_id = tuple(
            range(self.unused_warp_id + 1, self.unused_warp_id + self.num_convert_warps + 1)
        )
        self.threads_per_cta = 32 * len(
            (
                self.mma_warp_id,
                self.tma_warp_id,
                self.sched_warp_id,
                self.unused_warp_id,
                *self.convert_warp_id,
                *self.epilogue_warp_id,
            )
        )
        self.num_epilogue_warps = len(self.epilogue_warp_id)
        self.num_convert_threads = self.num_convert_warps * cute.arch.WARP_SIZE
        # Set barrier id for cta sync, epilogue sync and tmem ptr sync
        self.epilog_sync_bar_id = 1
        self.tmem_alloc_sync_bar_id = 2
        self.tmem_dealloc_sync_bar_id = 3

    def _create_tiled_mma(self):
        return utils.sm100.make_trivial_tiled_mma(
            self.mma_dtype,
            self.mma_dtype,
            self.a_major_mode,
            self.b_major_mode,
            self.acc_dtype,
            self.cta_group,
            self.mma_tiler[:2],
        )

    def _compute_stages(self) -> Tuple[int, int, int]:
        # Default C stages
        num_c_stage = 2

        c_bytes_per_stage = cute.size_in_bytes(self.c_dtype, self.c_smem_layout)

        mbar_helpers_bytes = 1024

        # Calculate smem layout and size for one stage of A, B, and C with 1-stage
        a_load_bytes_per_stage = cute.size_in_bytes(
            self.a_dtype,
            utils.sm100.make_smem_layout_a(self.tiled_mma, self.mma_tiler, self.a_dtype, 1),
        )
        b_load_bytes_per_stage = cute.size_in_bytes(
            self.b_dtype,
            utils.sm100.make_smem_layout_b(self.tiled_mma, self.mma_tiler, self.b_dtype, 1),
        )
        a_mma_bytes_per_stage = cute.size_in_bytes(
            self.mma_dtype,
            utils.sm100.make_smem_layout_a(self.tiled_mma, self.mma_tiler, self.mma_dtype, 1),
        )
        b_mma_bytes_per_stage = cute.size_in_bytes(
            self.mma_dtype,
            utils.sm100.make_smem_layout_b(self.tiled_mma, self.mma_tiler, self.mma_dtype, 1),
        )

        a_bytes_per_stage = a_load_bytes_per_stage + (
            a_mma_bytes_per_stage if self.do_transform_a else 0
        )
        b_bytes_per_stage = b_load_bytes_per_stage + (
            b_mma_bytes_per_stage if self.do_transform_b else 0
        )
        ab_bytes_per_stage = a_bytes_per_stage + b_bytes_per_stage

        num_a_stage = 0
        num_b_stage = 0

        def residual_smem_bytes():
            return (
                self.smem_capacity // self.occupancy
                - mbar_helpers_bytes
                - c_bytes_per_stage * num_c_stage
                - a_bytes_per_stage * num_a_stage
                - b_bytes_per_stage * num_b_stage
            )

        # Calculate A/B stages:
        # Assume same stage count for A/B load/mma buffers
        # Start with total smem per CTA (capacity / occupancy)
        # Subtract reserved bytes and initial C stages bytes
        # Divide remaining by bytes needed per A/B stage
        num_ab_stage = residual_smem_bytes() // ab_bytes_per_stage
        num_a_stage = num_ab_stage
        num_b_stage = num_ab_stage

        # Increase stages of transformed operand
        if self.do_transform_b:
            residual_stages = residual_smem_bytes() // b_bytes_per_stage
            num_b_stage += residual_stages
        if self.do_transform_a:
            residual_stages = residual_smem_bytes() // a_bytes_per_stage
            num_a_stage += residual_stages

        # Add remaining unused smem to epilogue
        residual_stages = residual_smem_bytes() // c_bytes_per_stage
        num_c_stage += residual_stages

        # Increase stages of non-transformed operands
        if not self.do_transform_b:
            residual_stages = residual_smem_bytes() // b_bytes_per_stage
            num_b_stage += residual_stages
        if not self.do_transform_a:
            residual_stages = residual_smem_bytes() // a_bytes_per_stage
            num_a_stage += residual_stages
            # If not converting, load and mma are same pipeline
        return num_a_stage, num_b_stage, num_c_stage

    def _setup_attributes(self):
        self.tiled_mma = self._create_tiled_mma()
        self.ctas_per_mma = cute.size(self.tiled_mma.thr_id.shape)
        # Compute mma/cluster/tile shapes
        mma_inst_shape_k = cute.size(self.tiled_mma.shape_mnk, mode=[2])
        mma_inst_tile_k = 4
        self.mma_tiler = (
            self.mma_tiler[0],
            self.mma_tiler[1],
            mma_inst_shape_k * mma_inst_tile_k,
        )
        self.cta_tile_shape_mnk = (
            self.mma_tiler[0] // self.ctas_per_mma,
            self.mma_tiler[1],
            self.mma_tiler[2],
        )

        # Compute cluster layout
        self.cluster_layout_vmnk = cute.tiled_divide(
            cute.make_layout((*self.cluster_shape_mn, 1)),
            (self.tiled_mma.thr_id.shape,),
        )

        # Compute number of multicast CTAs for A/B
        self.num_mcast_ctas_a = cute.size(self.cluster_layout_vmnk.shape[2])
        self.num_mcast_ctas_b = cute.size(self.cluster_layout_vmnk.shape[1])
        self.is_a_mcast = self.num_mcast_ctas_a > 1
        self.is_b_mcast = self.num_mcast_ctas_b > 1

        # Compute epilogue subtile
        self.epi_tile = utils.sm100.compute_epilogue_tile_shape(
            self.cta_tile_shape_mnk,
            self.use_2cta_instrs,
            self.c_layout,
            self.c_dtype,
        )

        self.c_smem_layout = utils.sm100.make_smem_layout_epi(
            self.c_dtype, self.c_layout, self.epi_tile, 1
        )

        self.smem_capacity = utils.get_smem_capacity_in_bytes()

        # Setup A/B/C stage count in shared memory and ACC stage count in tensor memory
        self.num_acc_stage = 2

        (
            self.num_a_stage,
            self.num_b_stage,
            self.num_c_stage,
        ) = self._compute_stages()

        # Setup clc stage by default
        self.num_clc_stage = 1
        assert self.num_clc_stage == 1, "Only single-stage CLC pipeline is supported"

        # Compute A/B/C shared memory layout
        # (CTA_ATOM, CTA_M, CTA_K, STAGE)
        self.a_load_smem_layout_staged = utils.sm100.make_smem_layout_a(
            self.tiled_mma, self.mma_tiler, self.a_dtype, self.num_a_stage
        )
        self.a_mma_smem_layout_staged = utils.sm100.make_smem_layout_a(
            self.tiled_mma, self.mma_tiler, self.mma_dtype, self.num_a_stage
        )
        # (CTA_ATOM, CTA_N, CTA_K, STAGE)
        self.b_load_smem_layout_staged = utils.sm100.make_smem_layout_b(
            self.tiled_mma, self.mma_tiler, self.b_dtype, self.num_b_stage
        )
        self.b_mma_smem_layout_staged = utils.sm100.make_smem_layout_b(
            self.tiled_mma, self.mma_tiler, self.mma_dtype, self.num_b_stage
        )

        self.c_smem_layout_staged = utils.sm100.make_smem_layout_epi(
            self.c_dtype, self.c_layout, self.epi_tile, self.num_c_stage
        )

        # Compute the number of tensor memory allocation columns
        self.num_tmem_alloc_cols = self._compute_num_tmem_alloc_cols(
            self.tiled_mma, self.mma_tiler, self.num_acc_stage, self.arch
        )

    @cute.jit
    def __call__(
        self,
        a: cute.Tensor,
        b: cute.Tensor,
        c: cute.Tensor,
        stream: cuda.CUstream,
        epilogue_op: cutlass.Constexpr[Callable | EpilogueFunction] = lambda x: x,
    ):
        # Permute from Torch to CuTe layout conventions
        # (l,m,k) -> (m,k,l)
        a = cute.make_tensor(a.iterator, cute.select(a.layout, mode=[1, 2, 0]))
        # (l,n,k) -> (n,k,l)
        b = cute.make_tensor(b.iterator, cute.select(b.layout, mode=[1, 2, 0]))
        # (l,m,n) -> (m,n,l)
        c = cute.make_tensor(c.iterator, cute.select(c.layout, mode=[1, 2, 0]))

        # Setup static attributes before smem/grid/tma computation
        assert self.a_dtype == a.element_type
        assert self.b_dtype == b.element_type
        assert self.c_dtype == c.element_type
        self.a_major_mode = utils.LayoutEnum.from_tensor(a).mma_major_mode()
        self.b_major_mode = utils.LayoutEnum.from_tensor(b).mma_major_mode()
        self.c_layout = utils.LayoutEnum.from_tensor(c)

        # Setup attributes that dependent on gemm inputs
        self._setup_attributes()

        # Setup TMA loads
        # For converted operand, consumers are convert threads on the same CTA, so
        # TMA load pipeline isn't CTA-pair-aware
        cta_group = tcgen05.CtaGroup.TWO if self.ctas_per_mma == 2 else tcgen05.CtaGroup.ONE
        cta_group_a = cta_group if not self.do_transform_a else tcgen05.CtaGroup.ONE
        if const_expr(self.is_a_mcast):
            a_op = cpasync.CopyBulkTensorTileG2SMulticastOp(cta_group_a)
        else:
            a_op = cpasync.CopyBulkTensorTileG2SOp(cta_group_a)
        cta_group_b = cta_group if not self.do_transform_b else tcgen05.CtaGroup.ONE
        if const_expr(self.is_b_mcast):
            b_op = cpasync.CopyBulkTensorTileG2SMulticastOp(cta_group_b)
        else:
            b_op = cpasync.CopyBulkTensorTileG2SOp(cta_group_b)

        a_load_smem_layout = cute.slice_(self.a_load_smem_layout_staged, (None, None, None, 0))
        tma_atom_a, tma_tensor_a = cute.nvgpu.make_tiled_tma_atom_A(
            a_op,
            a,
            a_load_smem_layout,
            self.mma_tiler,
            self.tiled_mma,
            self.cluster_layout_vmnk.shape,
            internal_type=(cutlass.TFloat32 if a.element_type is cutlass.Float32 else None),
        )

        b_load_smem_layout = cute.slice_(self.b_load_smem_layout_staged, (None, None, None, 0))
        tma_atom_b, tma_tensor_b = cute.nvgpu.make_tiled_tma_atom_B(
            b_op,
            b,
            b_load_smem_layout,
            self.mma_tiler,
            self.tiled_mma,
            self.cluster_layout_vmnk.shape,
            internal_type=(cutlass.TFloat32 if b.element_type is cutlass.Float32 else None),
        )

        a_mma_smem_layout = cute.slice_(self.a_mma_smem_layout_staged, (None, None, None, 0))
        b_mma_smem_layout = cute.slice_(self.b_mma_smem_layout_staged, (None, None, None, 0))
        self.a_load_bytes = cute.size_in_bytes(self.a_dtype, a_load_smem_layout) * (
            1 if self.do_transform_a else self.ctas_per_mma
        )
        self.b_load_bytes = cute.size_in_bytes(self.b_dtype, b_load_smem_layout) * (
            1 if self.do_transform_b else self.ctas_per_mma
        )
        # Response size is 4B * 4 elements
        self.num_clc_response_bytes = 16

        # Setup TMA store for C
        epi_smem_layout = cute.select(self.c_smem_layout_staged, mode=[0, 1])
        tma_atom_c, tma_tensor_c = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileS2GOp(), c, epi_smem_layout, self.epi_tile
        )
        c_smem_size = cute.cosize(self.c_smem_layout_staged)

        # Tiled Copies for dtype conversion
        max_elt_width_a = max(self.a_dtype.width, self.mma_dtype.width)
        vec_size_a = 128 // max_elt_width_a
        threads_per_row_a = min(16, self.mma_tiler[2] // vec_size_a)
        val_layout_a = cute.make_layout((1, vec_size_a))
        thr_layout_a = cute.make_ordered_layout(
            (self.num_convert_threads // threads_per_row_a, threads_per_row_a),
            order=(1, 0),
        )
        copy_atom_a_input = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(),
            self.a_dtype,
            num_bits_per_copy=vec_size_a * self.a_dtype.width,
        )
        copy_atom_a_transform = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(),
            self.mma_dtype,
            num_bits_per_copy=vec_size_a * self.mma_dtype.width,
        )
        tiled_copy_a_input = cute.make_tiled_copy_tv(
            copy_atom_a_input,
            thr_layout_a,
            val_layout_a,
        )
        tiled_copy_a_transform = cute.make_tiled_copy_tv(
            copy_atom_a_transform,
            thr_layout_a,
            val_layout_a,
        )

        max_elt_width_b = max(self.b_dtype.width, self.mma_dtype.width)
        vec_size_b = 128 // max_elt_width_b
        threads_per_row_b = min(16, self.mma_tiler[2] // vec_size_b)
        val_layout_b = cute.make_layout((1, vec_size_b))
        thr_layout_b = cute.make_ordered_layout(
            (self.num_convert_threads // threads_per_row_b, threads_per_row_b),
            order=(1, 0),
        )
        copy_atom_b_input = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(),
            self.b_dtype,
            num_bits_per_copy=vec_size_b * self.b_dtype.width,
        )
        copy_atom_b_transform = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(),
            self.mma_dtype,
            num_bits_per_copy=vec_size_b * self.mma_dtype.width,
        )
        tiled_copy_b_input = cute.make_tiled_copy_tv(
            copy_atom_b_input,
            thr_layout_b,
            val_layout_b,
        )
        tiled_copy_b_transform = cute.make_tiled_copy_tv(
            copy_atom_b_transform,
            thr_layout_b,
            val_layout_b,
        )

        # Set up SMEM
        self.smem_buffer_align_bytes = 1024

        @cute.struct
        class SharedStorage:
            a_load_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_a_stage * 2]
            b_load_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_b_stage * 2]
            a_mma_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_a_stage * 2]
            b_mma_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_b_stage * 2]
            acc_full_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_acc_stage * 2]
            tmem_dealloc_mbar: cutlass.Int64
            tmem_holding_buf: cutlass.Int32
            clc_mbar_ptr: cute.struct.MemRange[cutlass.Int64, 2]
            clc_response: cute.struct.MemRange[cutlass.Int32, 4]
            sA_load: cute.struct.Align[
                cute.struct.MemRange[self.a_dtype, cute.cosize(self.a_load_smem_layout_staged)],
                self.smem_buffer_align_bytes,
            ]
            sB_load: cute.struct.Align[
                cute.struct.MemRange[self.b_dtype, cute.cosize(self.b_load_smem_layout_staged)],
                self.smem_buffer_align_bytes,
            ]
            sA_mma: cute.struct.Align[
                cute.struct.MemRange[
                    self.mma_dtype,
                    cute.cosize(self.a_mma_smem_layout_staged) if self.do_transform_a else 0,
                ],
                self.smem_buffer_align_bytes,
            ]
            sB_mma: cute.struct.Align[
                cute.struct.MemRange[
                    self.mma_dtype,
                    cute.cosize(self.b_mma_smem_layout_staged) if self.do_transform_b else 0,
                ],
                self.smem_buffer_align_bytes,
            ]
            sC: cute.struct.Align[
                cute.struct.MemRange[self.c_dtype, c_smem_size],
                self.smem_buffer_align_bytes,
            ]

        self.shared_storage = SharedStorage

        if cutlass.const_expr(isinstance(epilogue_op, EpilogueFunction)):
            epilogue_op = epilogue_op.value

        # Set up tile scheduler and compute grid size
        c_tile_shape = cute.slice_(self.cta_tile_shape_mnk, (None, None, 0))
        gc = cute.zipped_divide(c, tiler=c_tile_shape)
        num_ctas_mnl = gc[(0, (None, None, None))].shape
        cluster_shape_mnl = (*self.cluster_shape_mn, 1)

        self.tile_sched_params = utils.ClcDynamicPersistentTileSchedulerParams(
            num_ctas_mnl,
            cluster_shape_mnl,
        )
        grid = utils.ClcDynamicPersistentTileScheduler.get_grid_shape(self.tile_sched_params)

        # Launch the kernel synchronously
        self.kernel(
            self.tiled_mma,
            tma_atom_a,
            tma_tensor_a,
            tma_atom_b,
            tma_tensor_b,
            tma_atom_c,
            tma_tensor_c,
            tiled_copy_a_input,
            tiled_copy_a_transform,
            tiled_copy_b_input,
            tiled_copy_b_transform,
            self.cluster_layout_vmnk,
            self.a_load_smem_layout_staged,
            self.b_load_smem_layout_staged,
            self.a_mma_smem_layout_staged,
            self.b_mma_smem_layout_staged,
            self.c_smem_layout_staged,
            self.epi_tile,
            self.tile_sched_params,
            epilogue_op,
        ).launch(
            grid=grid,
            block=[self.threads_per_cta, 1, 1],
            cluster=(*self.cluster_shape_mn, 1),
            stream=stream,
            smem=self.shared_storage.size_in_bytes(),
        )

    # GPU device kernel
    @cute.kernel
    def kernel(
        self,
        tiled_mma: cute.TiledMma,
        tma_atom_a: cute.CopyAtom,
        mA_mkl: cute.Tensor,
        tma_atom_b: cute.CopyAtom,
        mB_nkl: cute.Tensor,
        tma_atom_c: cute.CopyAtom,
        mC_mnl: cute.Tensor,
        tiled_copy_a_input: cute.TiledCopy,
        tiled_copy_a_transform: cute.TiledCopy,
        tiled_copy_b_input: cute.TiledCopy,
        tiled_copy_b_transform: cute.TiledCopy,
        cluster_layout_vmnk: cute.Layout,
        a_load_smem_layout_staged: cute.ComposedLayout,
        b_load_smem_layout_staged: cute.ComposedLayout,
        a_mma_smem_layout_staged: cute.ComposedLayout,
        b_mma_smem_layout_staged: cute.ComposedLayout,
        c_smem_layout_staged: Union[cute.Layout, cute.ComposedLayout, None],
        epi_tile: cute.Tile,
        tile_sched_params: utils.ClcDynamicPersistentTileSchedulerParams,
        epilogue_op: cutlass.Constexpr,
    ):
        """
        GPU device kernel performing the Persistent batched GEMM computation.
        """
        warp_idx = cute.arch.warp_idx()
        warp_idx = cute.arch.make_warp_uniform(warp_idx)

        #
        # Prefetch tma desc
        #
        if warp_idx == self.tma_warp_id:
            cpasync.prefetch_descriptor(tma_atom_a)
            cpasync.prefetch_descriptor(tma_atom_b)
            cpasync.prefetch_descriptor(tma_atom_c)

        use_2cta_instrs = cute.size(tiled_mma.thr_id.shape) == 2

        #
        # Setup cta/thread coordinates
        #
        # Coords inside cluster
        bidx, bidy, bidz = cute.arch.block_idx()
        mma_tile_coord_v = bidx % cute.size(tiled_mma.thr_id.shape)
        is_leader_cta = mma_tile_coord_v == 0
        cta_rank_in_cluster = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
        is_first_cta_in_cluster = cta_rank_in_cluster == 0
        block_in_cluster_coord_vmnk = cluster_layout_vmnk.get_flat_coord(cta_rank_in_cluster)
        # Coord inside cta
        tidx, _, _ = cute.arch.thread_idx()

        smem = utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)

        tma_load_pipeline_producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread)
        cta_v_size = cute.size(cluster_layout_vmnk, mode=[0])
        if const_expr(self.do_transform_a):
            # TMA load to dtype conversion pipeline: TMA warp is producer,
            # all epilogue warps from all mcast-partner CTAs are consumer
            num_tma_consumer_a = self.num_convert_warps * self.num_mcast_ctas_a
            a_load_pipeline_consumer_group = pipeline.CooperativeGroup(
                pipeline.Agent.Thread, num_tma_consumer_a
            )
            a_load_pipeline = pipeline.PipelineTmaAsync.create(
                barrier_storage=storage.a_load_mbar_ptr.data_ptr(),
                num_stages=self.num_a_stage,
                producer_group=tma_load_pipeline_producer_group,
                consumer_group=a_load_pipeline_consumer_group,
                tx_count=self.a_load_bytes,
                cta_layout_vmnk=cluster_layout_vmnk,
                mcast_mode_mn=(1, 0),  # only mcast in N direction
                tidx=tidx - (32 * self.convert_warp_id[0]),  # tidx in convert WG
                defer_sync=True,
            )

            # dtype conversion to MMA pipeline: epilogue warps are producer,
            # MMA warp is consumer. No inter-cluster operations.
            # TODO: syncwarp and have 1 thread arrive at producer barrier
            num_mma_producer = self.num_convert_warps * 32 * cta_v_size
            a_mma_pipeline_producer_group = pipeline.CooperativeGroup(
                pipeline.Agent.Thread, num_mma_producer
            )
            a_mma_pipeline_consumer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread)
            a_mma_pipeline = pipeline.PipelineAsyncUmma.create(
                barrier_storage=storage.a_mma_mbar_ptr.data_ptr(),
                num_stages=self.num_a_stage,
                producer_group=a_mma_pipeline_producer_group,
                consumer_group=a_mma_pipeline_consumer_group,
                cta_layout_vmnk=cluster_layout_vmnk,
                defer_sync=True,
            )
        else:
            # TMA load to MMA pipeline: TMA load warp is producer,
            # 1 thread from MMA warp from each mcast-partner CTA is consumer
            num_tma_consumer_a = self.num_mcast_ctas_a
            a_load_pipeline_consumer_group = pipeline.CooperativeGroup(
                pipeline.Agent.Thread, num_tma_consumer_a
            )
            a_load_pipeline = pipeline.PipelineTmaUmma.create(
                barrier_storage=storage.a_load_mbar_ptr.data_ptr(),
                num_stages=self.num_a_stage,
                producer_group=tma_load_pipeline_producer_group,
                consumer_group=a_load_pipeline_consumer_group,
                tx_count=self.a_load_bytes,
                cta_layout_vmnk=cluster_layout_vmnk,
                mcast_mode_mn=(1, 0),  # only mcast in N direction
                defer_sync=True,
            )

            # Alias MMA pipeline to load pipeline
            a_mma_pipeline = a_load_pipeline

        if const_expr(self.do_transform_b):
            # TMA load to dtype conversion pipeline: TMA warp is producer,
            # all epilogue warps from all mcast-partner CTAs are consumer
            num_tma_consumer_b = self.num_convert_warps * self.num_mcast_ctas_b
            b_load_pipeline_consumer_group = pipeline.CooperativeGroup(
                pipeline.Agent.Thread, num_tma_consumer_b
            )
            b_load_pipeline = pipeline.PipelineTmaAsync.create(
                barrier_storage=storage.b_load_mbar_ptr.data_ptr(),
                num_stages=self.num_b_stage,
                producer_group=tma_load_pipeline_producer_group,
                consumer_group=b_load_pipeline_consumer_group,
                tx_count=self.b_load_bytes,
                cta_layout_vmnk=cluster_layout_vmnk,
                mcast_mode_mn=(0, 1),  # only mcast in M direction
                tidx=tidx - (32 * self.convert_warp_id[0]),  # tidx in convert WG
                defer_sync=True,
            )

            # dtype conversion to MMA pipeline: epilogue warps are producer,
            # MMA warp is consumer. No multicast.
            # TODO: syncwarp and have 1 thread arrive at producer barrier
            num_mma_producer = self.num_convert_warps * 32 * cta_v_size
            b_mma_pipeline_producer_group = pipeline.CooperativeGroup(
                pipeline.Agent.Thread, num_mma_producer
            )
            b_mma_pipeline_consumer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread)
            b_mma_pipeline = pipeline.PipelineAsyncUmma.create(
                barrier_storage=storage.b_mma_mbar_ptr.data_ptr(),
                num_stages=self.num_b_stage,
                producer_group=b_mma_pipeline_producer_group,
                consumer_group=b_mma_pipeline_consumer_group,
                cta_layout_vmnk=cluster_layout_vmnk,
                defer_sync=True,
            )
        else:
            # TMA load to MMA pipeline: TMA load warp is producer,
            # 1 thread from MMA warp from each mcast-partner CTA is consumer
            num_tma_consumer_b = self.num_mcast_ctas_b
            b_load_pipeline_consumer_group = pipeline.CooperativeGroup(
                pipeline.Agent.Thread, num_tma_consumer_b
            )
            b_load_pipeline = pipeline.PipelineTmaUmma.create(
                barrier_storage=storage.b_load_mbar_ptr.data_ptr(),
                num_stages=self.num_b_stage,
                producer_group=tma_load_pipeline_producer_group,
                consumer_group=b_load_pipeline_consumer_group,
                tx_count=self.b_load_bytes,
                cta_layout_vmnk=cluster_layout_vmnk,
                mcast_mode_mn=(0, 1),  # only mcast in M direction
                defer_sync=True,
            )

            # Alias MMA pipeline to load pipeline
            b_mma_pipeline = b_load_pipeline

        # Initialize acc_pipeline (barrier) and states
        acc_pipeline_producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread)
        num_acc_consumer_threads = len(self.epilogue_warp_id) * cta_v_size
        acc_pipeline_consumer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread, num_acc_consumer_threads
        )
        acc_pipeline = pipeline.PipelineUmmaAsync.create(
            barrier_storage=storage.acc_full_mbar_ptr.data_ptr(),
            num_stages=self.num_acc_stage,
            producer_group=acc_pipeline_producer_group,
            consumer_group=acc_pipeline_consumer_group,
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        )

        # Initialize clc_pipeline (barrier) and states
        clc_pipeline_producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread)
        cluster_size = cute.size(self.cluster_shape_mn)
        num_clc_consumer_threads = 32 * (
            1 + cluster_size * (2 + len(self.epilogue_warp_id) + len(self.convert_warp_id))
        )
        clc_pipeline_consumer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread, num_clc_consumer_threads
        )
        clc_pipeline = pipeline.PipelineClcFetchAsync.create(
            barrier_storage=storage.clc_mbar_ptr.data_ptr(),
            num_stages=self.num_clc_stage,
            producer_group=clc_pipeline_producer_group,
            consumer_group=clc_pipeline_consumer_group,
            tx_count=self.num_clc_response_bytes,
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        )

        tmem_alloc_barrier = pipeline.NamedBarrier(
            barrier_id=self.tmem_alloc_sync_bar_id,
            num_threads=32 * len((self.mma_warp_id, *self.epilogue_warp_id)),
        )
        # Tensor memory dealloc barrier init
        tmem = utils.TmemAllocator(
            storage.tmem_holding_buf.ptr,
            barrier_for_retrieve=tmem_alloc_barrier,
            allocator_warp_id=self.epilogue_warp_id[0],
            is_two_cta=use_2cta_instrs,
            two_cta_tmem_dealloc_mbar_ptr=storage.tmem_dealloc_mbar.ptr,
        )

        # Cluster arrive after barrier init
        pipeline_init_arrive(cluster_shape_mn=cluster_layout_vmnk, is_relaxed=True)

        # Initial clc response pointer
        clc_response_ptr = storage.clc_response.data_ptr()

        clc_consumer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, self.num_clc_stage
        )

        #
        # Setup smem tensor A/B/C
        #
        # (CTA_ATOM, CTA_M, CTA_K, STAGE)
        sA_load = storage.sA_load.get_tensor(
            a_load_smem_layout_staged.outer,
            swizzle=a_load_smem_layout_staged.inner,
        )
        if const_expr(self.do_transform_a):
            sA_mma = storage.sA_mma.get_tensor(
                a_mma_smem_layout_staged.outer,
                swizzle=a_mma_smem_layout_staged.inner,
            )
        else:
            sA_mma = sA_load

        # (CTA_ATOM, CTA_N, CTA_K, STAGE)
        sB_load = storage.sB_load.get_tensor(
            b_load_smem_layout_staged.outer,
            swizzle=b_load_smem_layout_staged.inner,
        )
        if const_expr(self.do_transform_b):
            sB_mma = storage.sB_mma.get_tensor(
                b_mma_smem_layout_staged.outer,
                swizzle=b_mma_smem_layout_staged.inner,
            )
        else:
            sB_mma = sB_load

        sC = storage.sC.get_tensor(
            c_smem_layout_staged.outer,
            swizzle=c_smem_layout_staged.inner,
        )

        #
        # Compute multicast mask for A/B buffer full
        #
        a_full_mcast_mask = None
        b_full_mcast_mask = None
        if cutlass.const_expr(self.is_a_mcast or self.is_b_mcast or use_2cta_instrs):
            a_full_mcast_mask = cpasync.create_tma_multicast_mask(
                cluster_layout_vmnk, block_in_cluster_coord_vmnk, mcast_mode=2
            )
            b_full_mcast_mask = cpasync.create_tma_multicast_mask(
                cluster_layout_vmnk, block_in_cluster_coord_vmnk, mcast_mode=1
            )

        #
        # Local_tile partition global tensors
        #
        # (bM, bK, RestM, RestK, RestL)
        gA_mkl = cute.local_tile(
            mA_mkl, cute.slice_(self.mma_tiler, (None, 0, None)), (None, None, None)
        )
        # (bN, bK, RestN, RestK, RestL)
        gB_nkl = cute.local_tile(
            mB_nkl, cute.slice_(self.mma_tiler, (0, None, None)), (None, None, None)
        )
        # (bM, bN, RestM, RestN, RestL)
        gC_mnl = cute.local_tile(
            mC_mnl, cute.slice_(self.mma_tiler, (None, None, 0)), (None, None, None)
        )
        k_tile_cnt = cute.size(gA_mkl, mode=[3])

        #
        # Partition global tensor for TiledMMA_A/B/C
        #
        thr_mma = tiled_mma.get_slice(mma_tile_coord_v)
        # (MMA, MMA_M, MMA_K, RestM, RestK, RestL)
        tCgA = thr_mma.partition_A(gA_mkl)
        # (MMA, MMA_N, MMA_K, RestN, RestK, RestL)
        tCgB = thr_mma.partition_B(gB_nkl)
        # (MMA, MMA_M, MMA_N, RestM, RestN, RestL)
        tCgC = thr_mma.partition_C(gC_mnl)

        #
        # Partition global/shared tensor for TMA load A/B
        #
        # TMA load A partition_S/D
        a_cta_layout = cute.make_layout(cute.slice_(cluster_layout_vmnk, (0, 0, None, 0)).shape)
        # ((atom_v, rest_v), STAGE)
        # ((atom_v, rest_v), RestM, RestK, RestL)
        tAsA, tAgA = cpasync.tma_partition(
            tma_atom_a,
            block_in_cluster_coord_vmnk[2],
            a_cta_layout,
            cute.group_modes(sA_load, 0, 3),
            cute.group_modes(tCgA, 0, 3),
        )
        # TMA load B partition_S/D
        b_cta_layout = cute.make_layout(cute.slice_(cluster_layout_vmnk, (0, None, 0, 0)).shape)
        # ((atom_v, rest_v), STAGE)
        # ((atom_v, rest_v), RestM, RestK, RestL)
        tBsB, tBgB = cpasync.tma_partition(
            tma_atom_b,
            block_in_cluster_coord_vmnk[1],
            b_cta_layout,
            cute.group_modes(sB_load, 0, 3),
            cute.group_modes(tCgB, 0, 3),
        )

        #
        # Partition shared/tensor memory tensor for TiledMMA_A/B/C
        #
        # (MMA, MMA_M, MMA_K, STAGE)
        tCrA = tiled_mma.make_fragment_A(sA_mma)
        # (MMA, MMA_N, MMA_K, STAGE)
        tCrB = tiled_mma.make_fragment_B(sB_mma)
        # (MMA, MMA_M, MMA_N)
        acc_shape = tiled_mma.partition_shape_C(self.mma_tiler[:2])
        # (MMA, MMA_M, MMA_N, STAGE)
        tCtAcc_fake = tiled_mma.make_fragment_C(cute.append(acc_shape, self.num_acc_stage))

        #
        # Cluster wait before tensor memory alloc
        #
        pipeline_init_wait(cluster_shape_mn=cluster_layout_vmnk)

        #
        # Construct the scheduler
        #
        tile_sched = utils.ClcDynamicPersistentTileScheduler.create(
            tile_sched_params,
            cute.arch.block_idx(),
            cute.arch.grid_dim(),
            clc_response_ptr,
        )
        work_tile = tile_sched.initial_work_tile_info()

        #
        # Specialized TMA load warp
        #

        if warp_idx == self.tma_warp_id:
            a_load_producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.num_a_stage
            )
            b_load_producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.num_b_stage
            )

            #
            # Persistent tile scheduling loop
            #
            while work_tile.is_valid_tile:
                # Get tile coord from tile scheduler
                cur_tile_coord = work_tile.tile_idx
                mma_tile_coord_mnl = (
                    cur_tile_coord[0] // cute.size(tiled_mma.thr_id.shape),
                    cur_tile_coord[1],
                    cur_tile_coord[2],
                )

                #
                # Slice to per mma tile index
                #
                # ((atom_v, rest_v), RestK)
                tAgA_slice = tAgA[(None, mma_tile_coord_mnl[0], None, mma_tile_coord_mnl[2])]
                # ((atom_v, rest_v), RestK)
                tBgB_slice = tBgB[(None, mma_tile_coord_mnl[1], None, mma_tile_coord_mnl[2])]

                #
                # Tma load loop
                #
                for k_tile in cutlass.range(0, k_tile_cnt, 1, unroll=1):
                    if const_expr(self.do_transform_b and not self.do_transform_a):
                        # Load transformed matrix first

                        b_load_pipeline.producer_acquire(
                            b_load_producer_state,
                        )
                        cute.copy(
                            tma_atom_b,
                            tBgB_slice[(None, k_tile)],
                            tBsB[(None, b_load_producer_state.index)],
                            tma_bar_ptr=b_load_pipeline.producer_get_barrier(b_load_producer_state),
                            mcast_mask=b_full_mcast_mask,
                        )
                        b_load_pipeline.producer_commit(b_load_producer_state)
                        b_load_producer_state.advance()

                        a_load_pipeline.producer_acquire(
                            a_load_producer_state,
                        )

                        cute.copy(
                            tma_atom_a,
                            tAgA_slice[(None, k_tile)],
                            tAsA[(None, a_load_producer_state.index)],
                            tma_bar_ptr=a_load_pipeline.producer_get_barrier(a_load_producer_state),
                            mcast_mask=a_full_mcast_mask,
                        )
                        a_load_pipeline.producer_commit(a_load_producer_state)
                        a_load_producer_state.advance()

                    else:
                        a_load_pipeline.producer_acquire(
                            a_load_producer_state,
                        )
                        # TMA load A/B
                        cute.copy(
                            tma_atom_a,
                            tAgA_slice[(None, k_tile)],
                            tAsA[(None, a_load_producer_state.index)],
                            tma_bar_ptr=a_load_pipeline.producer_get_barrier(a_load_producer_state),
                            mcast_mask=a_full_mcast_mask,
                        )
                        a_load_pipeline.producer_commit(a_load_producer_state)
                        a_load_producer_state.advance()

                        b_load_pipeline.producer_acquire(
                            b_load_producer_state,
                        )
                        cute.copy(
                            tma_atom_b,
                            tBgB_slice[(None, k_tile)],
                            tBsB[(None, b_load_producer_state.index)],
                            tma_bar_ptr=b_load_pipeline.producer_get_barrier(b_load_producer_state),
                            mcast_mask=b_full_mcast_mask,
                        )
                        b_load_pipeline.producer_commit(b_load_producer_state)
                        b_load_producer_state.advance()

                #
                # Advance to next tile
                #
                clc_pipeline.consumer_wait(clc_consumer_state)
                work_tile = tile_sched.get_current_work()
                clc_pipeline.consumer_release(clc_consumer_state)
                clc_consumer_state.advance()
            #
            # Wait A/B buffer empty
            #
            a_load_pipeline.producer_tail(a_load_producer_state)
            b_load_pipeline.producer_tail(b_load_producer_state)

        #
        # Sched warp
        #
        elif warp_idx == self.sched_warp_id and is_first_cta_in_cluster:
            #
            # Persistent tile scheduling loop
            #
            clc_producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.ProducerConsumer, self.num_clc_stage
            )

            while work_tile.is_valid_tile:
                #
                # Advance to next tile
                #

                clc_pipeline.producer_acquire(clc_producer_state)
                mbarrier_addr = clc_pipeline.producer_get_barrier(clc_producer_state)
                tile_sched.advance_to_next_work(mbarrier_addr)
                clc_producer_state.advance()

                clc_pipeline.consumer_wait(clc_consumer_state)
                work_tile = tile_sched.get_current_work()

                clc_pipeline.consumer_release(clc_consumer_state)
                clc_consumer_state.advance()
            clc_pipeline.producer_tail(clc_producer_state)

        #
        # Specialized MMA warp
        #
        elif warp_idx == self.mma_warp_id:
            #
            # Retrieving tensor memory ptr and make accumulator tensor
            #
            tmem.wait_for_alloc()
            tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)
            # (MMA, MMA_M, MMA_N, STAGE)
            tCtAcc_base = cute.make_tensor(tmem_ptr, tCtAcc_fake.layout)

            #
            # Persistent tile scheduling loop
            #
            acc_producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.num_acc_stage
            )
            a_mma_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.num_a_stage
            )
            b_mma_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.num_b_stage
            )

            while work_tile.is_valid_tile:
                # Get tile coord from tile scheduler
                cur_tile_coord = work_tile.tile_idx
                mma_tile_coord_mnl = (
                    cur_tile_coord[0] // cute.size(tiled_mma.thr_id.shape),
                    cur_tile_coord[1],
                    cur_tile_coord[2],
                )

                # Set tensor memory buffer for current tile
                # (MMA, MMA_M, MMA_N)
                tCtAcc = tCtAcc_base[(None, None, None, acc_producer_state.index)]

                #
                # Mma mainloop
                #
                if is_leader_cta:
                    acc_pipeline.producer_acquire(acc_producer_state)
                    tiled_mma.set(tcgen05.Field.ACCUMULATE, False)
                    for k_tile in range(k_tile_cnt):
                        # Conditionally wait for AB buffer full
                        a_mma_pipeline.consumer_wait(
                            a_mma_consumer_state,
                        )
                        b_mma_pipeline.consumer_wait(
                            b_mma_consumer_state,
                        )

                        # tCtAcc += tCrA * tCrB
                        num_kblocks = cute.size(tCrA, mode=[2])
                        for kblk_idx in cutlass.range(num_kblocks, unroll_full=True):
                            cute.gemm(
                                tiled_mma,
                                tCtAcc,
                                tCrA[(None, None, kblk_idx, a_mma_consumer_state.index)],
                                tCrB[(None, None, kblk_idx, b_mma_consumer_state.index)],
                                tCtAcc,
                            )
                            # Enable accumulate on tCtAcc after first kblock
                            tiled_mma.set(tcgen05.Field.ACCUMULATE, True)

                        # Async arrive AB buffer empty
                        a_mma_pipeline.consumer_release(a_mma_consumer_state)
                        a_mma_consumer_state.advance()
                        b_mma_pipeline.consumer_release(b_mma_consumer_state)
                        b_mma_consumer_state.advance()

                    acc_pipeline.producer_commit(acc_producer_state)
                acc_producer_state.advance()

                #
                # Advance to next tile
                #
                clc_pipeline.consumer_wait(clc_consumer_state)
                work_tile = tile_sched.get_current_work()
                clc_pipeline.consumer_release(clc_consumer_state)
                clc_consumer_state.advance()
            #
            # Wait for accumulator buffer empty
            #
            acc_pipeline.producer_tail(acc_producer_state)

        elif warp_idx in self.convert_warp_id:
            convert_tidx = tidx - 32 * self.convert_warp_id[0]

            # Squash SMEM tile layout to (MN, K, STAGE)
            @cute.jit
            def squash_tensor_for_convert(
                t: cute.Tensor,
                is_a: cutlass.Constexpr[bool],
            ):
                tile_size_mn = self.mma_tiler[0 if is_a else 1] // self.ctas_per_mma
                tile_size_k = self.mma_tiler[2]
                num_stage = self.num_a_stage if is_a else self.num_b_stage
                return cute.composition(
                    t,
                    cute.make_ordered_layout(
                        (
                            tile_size_mn,
                            tile_size_k,
                            num_stage,
                        ),
                        order=(0, 1, 2),
                    ),
                )

            if const_expr(self.do_transform_a):
                a_load_consumer_state = pipeline.make_pipeline_state(
                    pipeline.PipelineUserType.Consumer, self.num_a_stage
                )
                a_mma_producer_state = pipeline.make_pipeline_state(
                    pipeline.PipelineUserType.Producer, self.num_a_stage
                )

                sA_load_for_convert = squash_tensor_for_convert(sA_load, True)
                sA_mma_for_convert = squash_tensor_for_convert(sA_mma, True)

                thr_copy_a_input = tiled_copy_a_input.get_slice(convert_tidx)
                # (COPY_ATOM, COPY_M, COPY_K, STAGE)
                tAsA_input = thr_copy_a_input.partition_S(sA_load_for_convert)
                # (COPY_ATOM, COPY, STAGE)
                tAsA_input = cute.group_modes(tAsA_input, 1, cute.rank(tAsA_input) - 1)
                tArA_input = cute.make_rmem_tensor(tAsA_input[(None, None, 0)].shape, self.a_dtype)

                thr_copy_a_transform = tiled_copy_a_transform.get_slice(convert_tidx)
                tAsA_transform = thr_copy_a_transform.partition_D(sA_mma_for_convert)
                tAsA_transform = cute.group_modes(tAsA_transform, 1, cute.rank(tAsA_transform) - 1)
                tArA_transform = cute.make_rmem_tensor(
                    tAsA_transform[(None, None, 0)].shape, self.mma_dtype
                )
            if const_expr(self.do_transform_b):
                b_load_consumer_state = pipeline.make_pipeline_state(
                    pipeline.PipelineUserType.Consumer, self.num_b_stage
                )
                b_mma_producer_state = pipeline.make_pipeline_state(
                    pipeline.PipelineUserType.Producer, self.num_b_stage
                )

                sB_load_for_convert = squash_tensor_for_convert(sB_load, False)
                sB_mma_for_convert = squash_tensor_for_convert(sB_mma, False)

                thr_copy_b_input = tiled_copy_b_input.get_slice(convert_tidx)
                # (COPY_ATOM, COPY_N, COPY_K, STAGE)
                tBsB_input = thr_copy_b_input.partition_S(sB_load_for_convert)
                # (COPY_ATOM, COPY, STAGE)
                tBsB_input = cute.group_modes(tBsB_input, 1, cute.rank(tBsB_input) - 1)
                tBrB_input = cute.make_rmem_tensor(tBsB_input[(None, None, 0)].shape, self.b_dtype)

                thr_copy_b_transform = tiled_copy_b_transform.get_slice(convert_tidx)
                tBsB_transform = thr_copy_b_transform.partition_D(sB_mma_for_convert)
                tBsB_transform = cute.group_modes(tBsB_transform, 1, cute.rank(tBsB_transform) - 1)
                tBrB_transform = cute.make_rmem_tensor(
                    tBsB_transform[(None, None, 0)].shape, self.mma_dtype
                )

            while work_tile.is_valid_tile:
                # Get tile coord from tile scheduler
                cur_tile_coord = work_tile.tile_idx
                mma_tile_coord_mnl = (
                    cur_tile_coord[0] // cute.size(tiled_mma.thr_id.shape),
                    cur_tile_coord[1],
                    cur_tile_coord[2],
                )
                num_tiles_executed = tile_sched.num_tiles_executed

                for k_tile in cutlass.range(0, k_tile_cnt, 1, unroll=1):
                    if const_expr(self.do_transform_a):
                        # Wait for TMA load
                        a_load_pipeline.consumer_wait(
                            a_load_consumer_state,
                        )
                        a_mma_pipeline.producer_acquire(
                            a_mma_producer_state,
                        )

                        tAsA_input_slice = tAsA_input[(None, None, a_load_consumer_state.index)]
                        tAsA_transform_slice = tAsA_transform[
                            (None, None, a_mma_producer_state.index)
                        ]

                        for idx in cutlass.range_constexpr(cute.size(tArA_input, mode=[1])):
                            # Load A from shared memory
                            cute.autovec_copy(
                                tAsA_input[(None, idx, a_load_consumer_state.index)],
                                tArA_input[(None, idx)],
                            )
                            # Convert it to mma dtype
                            tArA_transform[(None, idx)].store(
                                tArA_input[(None, idx)].load().to(self.mma_dtype)
                            )
                        # Store back to shared memory
                        cute.autovec_copy(
                            tArA_transform, tAsA_transform[(None, None, a_mma_producer_state.index)]
                        )

                    if const_expr(self.do_transform_b):
                        # Wait for TMA load
                        b_load_pipeline.consumer_wait(
                            b_load_consumer_state,
                        )
                        b_mma_pipeline.producer_acquire(
                            b_mma_producer_state,
                        )
                        tBsB_input_slice = tBsB_input[(None, None, b_load_consumer_state.index)]
                        tBsB_transform_slice = tBsB_transform[
                            (None, None, b_mma_producer_state.index)
                        ]

                        for idx in cutlass.range_constexpr(cute.size(tBrB_input, mode=[1])):
                            # Load B from shared memory
                            cute.autovec_copy(
                                tBsB_input[(None, idx, b_load_consumer_state.index)],
                                tBrB_input[(None, idx)],
                            )
                            # Convert it to mma dtype
                            tBrB_transform[(None, idx)].store(
                                tBrB_input[(None, idx)].load().to(self.mma_dtype)
                            )
                        # Store back to shared memory
                        cute.autovec_copy(
                            tBrB_transform, tBsB_transform[(None, None, b_mma_producer_state.index)]
                        )

                    if const_expr(self.do_transform_a or self.do_transform_b):
                        # Async fence to make SMEM store visible to UMMA
                        cute.arch.fence_proxy(
                            "async.shared",
                            space="cta",
                        )

                    if const_expr(self.do_transform_a):
                        a_load_pipeline.consumer_release(a_load_consumer_state)
                        a_mma_pipeline.producer_commit(a_mma_producer_state)
                        a_mma_producer_state.advance()
                        a_load_consumer_state.advance()
                    if const_expr(self.do_transform_b):
                        b_load_pipeline.consumer_release(b_load_consumer_state)
                        b_mma_pipeline.producer_commit(b_mma_producer_state)
                        b_mma_producer_state.advance()
                        b_load_consumer_state.advance()
                clc_pipeline.consumer_wait(clc_consumer_state)
                work_tile = tile_sched.get_current_work()
                clc_pipeline.consumer_release(clc_consumer_state)
                clc_consumer_state.advance()

            if const_expr(self.do_transform_a):
                a_mma_pipeline.producer_tail(a_mma_producer_state)
            if const_expr(self.do_transform_b):
                b_mma_pipeline.producer_tail(b_mma_producer_state)

        elif warp_idx in self.epilogue_warp_id:
            #
            # Alloc tensor memory buffer
            #
            tmem.allocate(self.num_tmem_alloc_cols)

            #
            # Retrieving tensor memory ptr and make accumulator tensor
            #
            tmem.wait_for_alloc()
            tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)
            # (MMA, MMA_M, MMA_N, STAGE)
            tCtAcc_base = cute.make_tensor(tmem_ptr, tCtAcc_fake.layout)

            acc_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.num_acc_stage
            )
            c_producer_group = pipeline.CooperativeGroup(
                pipeline.Agent.Thread,
                32 * len(self.epilogue_warp_id),
            )
            c_pipeline = pipeline.PipelineTmaStore.create(
                num_stages=self.num_c_stage, producer_group=c_producer_group
            )

            while work_tile.is_valid_tile:
                # Get tile coord from tile scheduler
                cur_tile_coord = work_tile.tile_idx
                mma_tile_coord_mnl = (
                    cur_tile_coord[0] // cute.size(tiled_mma.thr_id.shape),
                    cur_tile_coord[1],
                    cur_tile_coord[2],
                )
                num_tiles_executed = tile_sched.num_tiles_executed

                #
                # Epilogue
                #
                acc_consumer_state = self.epilogue_tma_store(
                    tidx,
                    warp_idx,
                    tma_atom_c,
                    tCtAcc_base,
                    sC,
                    tCgC,
                    epi_tile,
                    num_tiles_executed,
                    epilogue_op,
                    mma_tile_coord_mnl,
                    acc_consumer_state,
                    acc_pipeline,
                    c_pipeline,
                )
                #
                # Advance to next tile
                #
                clc_pipeline.consumer_wait(clc_consumer_state)
                work_tile = tile_sched.get_current_work()
                clc_pipeline.consumer_release(clc_consumer_state)
                clc_consumer_state.advance()

            c_pipeline.producer_tail()
            #
            # Dealloc the tensor memory buffer
            #
            tmem.relinquish_alloc_permit()
            tmem.free(tmem_ptr)

    @cute.jit
    def epilogue_tma_store(
        self,
        epi_tidx: Int32,
        warp_idx: Int32,
        tma_atom_c: cute.CopyAtom,
        # Input of epilogue
        tCtAcc_base: cute.Tensor,
        # Staging of epilogue
        sC: cute.Tensor,
        # Output of epilogue
        tCgC_base: cute.Tensor,
        epi_tile: cute.Tile,
        num_tiles_executed: Int32,
        epilogue_op: Constexpr,
        mma_tile_coord_mnl: Tuple[Int32, Int32, Int32],
        acc_consumer_state: pipeline.PipelineState,
        acc_pipeline: pipeline.PipelineAsync,
        c_pipeline: pipeline.PipelineTmaStore,
    ) -> pipeline.PipelineState:
        # Layout transformation for tCgC_base
        # ((MMA_ATOM_M, MMA_ATOM_N), MMA_M, MMA_N, TILE_M, TILE_N, TILE_K)
        # -> ((MMA_ATOM_M, MMA_M), (MMA_ATOM_N, MMA_N), TILE_M, TILE_N, TILE_K)
        tCgC = transform_partitioned_tensor_layout(tCgC_base)

        # Layout transformation for tCtAcc_base
        # ((MMA_ATOM_M, MMA_ATOM_N), MMA_M, MMA_N, STAGE)
        # -> ((MMA_ATOM_M, MMA_M), (MMA_ATOM_N, MMA_N), STAGE)
        tCtAcc = transform_partitioned_tensor_layout(tCtAcc_base)

        tiled_copy_t2r, tTR_tAcc_base, tTR_rAcc = epilogue_tmem_copy_and_partition(
            self, epi_tidx, tCtAcc, tCgC, epi_tile, self.use_2cta_instrs
        )

        tTR_rC = cute.make_rmem_tensor(tTR_rAcc.shape, self.c_dtype)
        tiled_copy_r2s, tRS_rC, tRS_sC = epilogue_smem_copy_and_partition(
            self, tiled_copy_t2r, tTR_rC, epi_tidx, sC
        )

        # (EPI_TILE_M, EPI_TILE_N, EPI_M, EPI_N, RestM, RestN, RestL)
        tCgC_epi = cute.flat_divide(tCgC, epi_tile)
        # ((ATOM_V, REST_V), EPI_M, EPI_N)
        # ((ATOM_V, REST_V), EPI_M, EPI_N, RestM, RestN, RestL)
        bSG_sC, bSG_gC_partitioned = cpasync.tma_partition(
            tma_atom_c,
            0,
            cute.make_layout(1),
            cute.group_modes(sC, 0, 2),
            cute.group_modes(tCgC_epi, 0, 2),
        )

        epilog_sync_barrier = pipeline.NamedBarrier(
            barrier_id=self.epilog_sync_bar_id,
            num_threads=32 * len(self.epilogue_warp_id),
        )

        #
        # Slice to per mma tile index
        #
        # ((ATOM_V, REST_V), EPI_M, EPI_N)
        bSG_gC = bSG_gC_partitioned[(None, None, None, *mma_tile_coord_mnl)]

        # Set tensor memory buffer for current tile
        # (T2R, T2R_M, T2R_N, EPI_M, EPI_M)
        tTR_tAcc = tTR_tAcc_base[(None, None, None, None, None, acc_consumer_state.index)]

        #
        # Wait for accumulator buffer full
        #
        acc_pipeline.consumer_wait(acc_consumer_state)

        tTR_tAcc = cute.group_modes(tTR_tAcc, 3, cute.rank(tTR_tAcc))
        bSG_gC = cute.group_modes(bSG_gC, 1, cute.rank(bSG_gC))

        #
        # Store accumulator to global memory in subtiles
        #
        subtile_cnt = cute.size(tTR_tAcc.shape, mode=[3])  # type: ignore[union-attr]
        num_prev_subtiles = num_tiles_executed * subtile_cnt
        for subtile_idx in range(subtile_cnt):
            #
            # Load accumulator from tensor memory buffer to register
            #
            tTR_tAcc_mn = tTR_tAcc[(None, None, None, subtile_idx)]  # type: ignore[call-overload]
            cute.copy(tiled_copy_t2r, tTR_tAcc_mn, tTR_rAcc)

            #
            # Convert to C type
            #
            acc_vec = tiled_copy_r2s.retile(tTR_rAcc).load()
            acc_vec = epilogue_op(acc_vec).to(self.c_dtype)
            tRS_rC.store(acc_vec)

            #
            # Store C to shared memory
            #
            c_buffer = (num_prev_subtiles + subtile_idx) % self.num_c_stage
            cute.copy(tiled_copy_r2s, tRS_rC, tRS_sC[(None, None, None, c_buffer)])
            # Fence and barrier to make sure shared memory store is visible to TMA store
            cute.arch.fence_proxy(
                "async.shared",
                space="cta",
            )
            epilog_sync_barrier.arrive_and_wait()

            #
            # TMA store C to global memory
            #
            if warp_idx == self.epilogue_warp_id[0]:
                cute.copy(
                    tma_atom_c,
                    bSG_sC[(None, c_buffer)],
                    bSG_gC[(None, subtile_idx)],
                )
                # Fence and barrier to make sure shared memory store is visible to TMA store
                c_pipeline.producer_commit()
                c_pipeline.producer_acquire()
            epilog_sync_barrier.arrive_and_wait()

        epilog_sync_barrier.arrive_and_wait()

        #
        # Async arrive accumulator buffer empty
        #
        with cute.arch.elect_one():
            acc_pipeline.consumer_release(acc_consumer_state)
        acc_consumer_state.advance()
        return acc_consumer_state

    @staticmethod
    def _compute_num_tmem_alloc_cols(
        tiled_mma: cute.TiledMma,
        mma_tiler: Tuple[int, int, int],
        num_acc_stage: int,
        arch: str,
    ) -> int:
        """
        Compute the number of tensor memory allocation columns.

        :param tiled_mma: The tiled MMA object defining the core computation.
        :type tiled_mma: cute.TiledMma
        :param mma_tiler: The shape (M, N, K) of the MMA tile.
        :type mma_tiler: tuple[int, int, int]
        :param num_acc_stage: The stage of the accumulator tensor.
        :type num_acc_stage: int

        :return: The number of tensor memory allocation columns.
        :rtype: int
        """
        acc_shape = tiled_mma.partition_shape_C(mma_tiler[:2])
        tCtAcc_fake = tiled_mma.make_fragment_C(cute.append(acc_shape, num_acc_stage))
        num_tmem_alloc_cols = utils.get_num_tmem_alloc_cols(tCtAcc_fake, arch=arch)

        return num_tmem_alloc_cols

    def check_supported_dtypes(self):
        valid_ab_dtypes = {
            cutlass.Float16,
            cutlass.BFloat16,
            cutlass.Float32,
        }
        if self.a_dtype not in valid_ab_dtypes or self.b_dtype not in valid_ab_dtypes:
            raise testing.CantImplementError(
                f"Unsupported AB load dtype: {self.a_dtype} and {self.b_dtype}"
            )

        if self.acc_dtype not in {cutlass.Float32, cutlass.Float16}:
            raise testing.CantImplementError(f"Unsupported accumulator dtype: {self.acc_dtype}")

        if self.mma_dtype not in {cutlass.TFloat32, cutlass.BFloat16, cutlass.Float16}:
            raise testing.CantImplementError(f"Unsupported AB MMA dtype: {self.mma_dtype}")

        # Define compatibility mapping between accumulator type and AB type
        acc_mma_compatibility = {
            cutlass.Float32: {
                cutlass.TFloat32,
                cutlass.Float16,
                cutlass.BFloat16,
            },
            cutlass.Float16: {
                cutlass.Float16,
            },
        }
        # Check compatibility between accumulator type and AB MMA type
        if self.mma_dtype not in acc_mma_compatibility[self.acc_dtype]:
            raise testing.CantImplementError(
                f"Unsupported MMA dtype: {self.mma_dtype} for accumulator dtype: {self.acc_dtype}"
            )

        # Define compatibility mapping between accumulator type and C type
        acc_c_compatibility = {
            cutlass.Float32: {
                cutlass.Float32,
                cutlass.Float16,
                cutlass.BFloat16,
            },
            cutlass.Float16: {
                cutlass.BFloat16,
                cutlass.Float16,
            },
        }
        # Check compatibility between accumulator type and C type
        if self.c_dtype not in acc_c_compatibility[self.acc_dtype]:
            raise testing.CantImplementError(
                f"Unsupported C dtype: {self.c_dtype} for accumulator dtype: {self.acc_dtype}"
            )

    def check_mma_tiler_and_cluster_shape(self):
        """Check if the mma tiler and cluster shape are valid.

        :raises testing.CantImplementError: If the mma tiler and cluster shape are invalid
        """
        # Skip invalid mma tile shape
        if not (
            (not self.use_2cta_instrs and self.mma_tiler_mn[0] in [64, 128])
            or (self.use_2cta_instrs and self.mma_tiler_mn[0] in [128, 256])
        ):
            raise testing.CantImplementError(
                f"Invalid mma tiler & use_2cta_instrs: {self.mma_tiler_mn}, {self.use_2cta_instrs}"
            )
        if self.mma_tiler_mn[1] not in range(32, 257, 32):
            raise testing.CantImplementError(f"Invalid mma tiler N: {self.mma_tiler_mn[1]}")
        # Skip illegal cluster shape
        if self.cluster_shape_mn[0] % (2 if self.use_2cta_instrs else 1) != 0:
            raise testing.CantImplementError(f"Invalid cluster shape M: {self.cluster_shape_mn[0]}")
        # Skip invalid cluster shape
        is_power_of_2 = lambda x: x > 0 and (x & (x - 1)) == 0
        if (
            self.cluster_shape_mn[0] * self.cluster_shape_mn[1] > 16
            or self.cluster_shape_mn[0] <= 0
            or self.cluster_shape_mn[1] <= 0
            or not is_power_of_2(self.cluster_shape_mn[0])
            or not is_power_of_2(self.cluster_shape_mn[1])
        ):
            raise testing.CantImplementError(f"Invalid cluster shape: {self.cluster_shape_mn}")

    def check_can_implement(self):
        """
        Determine if the given tensor configuration can be implemented by this kernel.
        :raises CantImplementError: if the kernel can't be implemented
        """

        # try:
        # Skip unsupported types
        self.check_supported_dtypes()

        # Skip invalid mma tile shape and cluster shape
        self.check_mma_tiler_and_cluster_shape()
