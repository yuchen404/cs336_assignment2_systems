"""
Triton 教程：基于 weighted_sum 例子学习 Triton 编程
"""

import torch
import triton
import triton.language as tl


# ==============================================================================
# 例子 1: 加权求和（你的原始例子，带详细注释）
# ==============================================================================

@triton.jit
def weighted_sum_fwd(
    # === 输入输出指针 ===
    x_ptr,          # 输入矩阵 x 的指针，形状 (ROWS, D)
    weight_ptr,     # 权重向量的指针，形状 (D,)
    output_ptr,     # 输出向量的指针，形状 (ROWS,)
    
    # === 步幅信息 ===
    # 步幅告诉我们如何在内存中移动
    x_stride_row,   # x 每行之间的距离（通常是 D）
    x_stride_dim,   # x 每列之间的距离（通常是 1）
    weight_stride_dim,   # weight 每个元素的距离（通常是 1）
    output_stride_row,   # output 每个元素的距离（通常是 1）
    
    # === 形状信息 ===
    ROWS,           # x 的行数
    D,              # x 的列数（特征维度）
    
    # === 编译时常量 ===
    # tl.constexpr 表示这些值在编译时必须已知
    ROWS_TILE_SIZE: tl.constexpr,  # 每次处理多少行
    D_TILE_SIZE: tl.constexpr,     # 每次处理多少列
):
    """
    计算: output[i] = sum(x[i, :] * weight[:])
    
    核心思想：
    - 每个程序实例处理 ROWS_TILE_SIZE 行
    - 沿着 D 维度分块处理，每次处理 D_TILE_SIZE 个元素
    """
    
    # === 步骤 1: 确定当前程序实例负责的行范围 ===
    row_tile_idx = tl.program_id(0)  # 第几个块（0, 1, 2, ...）
    # 该实例负责行 [row_tile_idx * ROWS_TILE_SIZE, (row_tile_idx + 1) * ROWS_TILE_SIZE)
    
    # === 步骤 2: 创建块指针 ===
    # 块指针是 Triton 的高级抽象，简化了多维张量访问
    x_block_ptr = tl.make_block_ptr(
        base=x_ptr,
        shape=(ROWS, D),                    # 告诉 Triton 总形状，用于边界检查
        strides=(x_stride_row, x_stride_dim),  # 告诉 Triton 内存布局
        offsets=(row_tile_idx * ROWS_TILE_SIZE, 0),  # 从哪里开始读取
        block_shape=(ROWS_TILE_SIZE, D_TILE_SIZE),   # 每次读取多大的块
        order=(1, 0),  # 列主序优化提示 (0=行主, 1=列主)
    )
    
    weight_block_ptr = tl.make_block_ptr(
        base=weight_ptr,
        shape=(D,),
        strides=(weight_stride_dim,),
        offsets=(0,),  # weight 从头开始
        block_shape=(D_TILE_SIZE,),
        order=(0,),
    )
    
    output_block_ptr = tl.make_block_ptr(
        base=output_ptr,
        shape=(ROWS,),
        strides=(output_stride_row,),
        offsets=(row_tile_idx * ROWS_TILE_SIZE,),
        block_shape=(ROWS_TILE_SIZE,),
        order=(0,),
    )
    
    # === 步骤 3: 初始化累加器 ===
    output = tl.zeros((ROWS_TILE_SIZE,), dtype=tl.float32)
    
    # === 步骤 4: 沿 D 维度分块计算 ===
    # tl.cdiv(D, D_TILE_SIZE) 是向上取整的除法
    for i in range(tl.cdiv(D, D_TILE_SIZE)):
        # 加载当前块
        # boundary_check=(0, 1) 表示在第0维和第1维都检查边界
        # padding_option="zero" 表示越界位置填充0
        row = tl.load(x_block_ptr, boundary_check=(0, 1), padding_option="zero")
        # row 形状: (ROWS_TILE_SIZE, D_TILE_SIZE)
        
        weight = tl.load(weight_block_ptr, boundary_check=(0,), padding_option="zero")
        # weight 形状: (D_TILE_SIZE,)
        
        # 计算加权和
        # weight[None, :] 将 (D_TILE_SIZE,) 广播为 (1, D_TILE_SIZE)
        # row * weight[None, :] 得到 (ROWS_TILE_SIZE, D_TILE_SIZE)
        # tl.sum(..., axis=1) 沿列求和，得到 (ROWS_TILE_SIZE,)
        output += tl.sum(row * weight[None, :], axis=1)
        
        # 移动指针到下一个块
        x_block_ptr = tl.block_ptr.advance(x_block_ptr, (0, D_TILE_SIZE))
        weight_block_ptr = tl.block_ptr.advance(weight_block_ptr, (D_TILE_SIZE,))
    
    # === 步骤 5: 写回结果 ===
    tl.store(output_block_ptr, output, boundary_check=(0,))


# Python 包装函数
def weighted_sum(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """
    x: (ROWS, D)
    weight: (D,)
    返回: (ROWS,)
    """
    ROWS, D = x.shape
    assert weight.shape == (D,), f"Weight shape mismatch: {weight.shape} vs ({D},)"
    
    output = torch.empty(ROWS, device=x.device, dtype=x.dtype)
    
    # 选择块大小（调优参数）
    ROWS_TILE_SIZE = 32
    D_TILE_SIZE = 128
    
    # 启动内核
    grid = (triton.cdiv(ROWS, ROWS_TILE_SIZE),)  # 需要多少个程序实例
    
    weighted_sum_fwd[grid](
        x, weight, output,
        x.stride(0), x.stride(1),
        weight.stride(0),
        output.stride(0),
        ROWS, D,
        ROWS_TILE_SIZE, D_TILE_SIZE,
    )
    
    return output


# ==============================================================================
# 例子 2: 向量加法（更简单的入门例子）
# ==============================================================================

@triton.jit
def vector_add_kernel(
    x_ptr, y_ptr, output_ptr,
    N,
    BLOCK_SIZE: tl.constexpr,
):
    """最简单的 Triton 内核：向量加法"""
    # 1. 计算当前块负责的索引范围
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    
    # 2. 生成偏移量
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # 3. 创建掩码（处理边界）
    mask = offsets < N
    
    # 4. 加载数据
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    
    # 5. 计算
    output = x + y
    
    # 6. 存储结果
    tl.store(output_ptr + offsets, output, mask=mask)


def vector_add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    N = x.shape[0]
    output = torch.empty_like(x)
    
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(N, BLOCK_SIZE),)
    
    vector_add_kernel[grid](x, y, output, N, BLOCK_SIZE)
    return output


# ==============================================================================
# 例子 3: 矩阵乘法（展示 2D 分块）
# ==============================================================================

@triton.jit
def matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    """
    计算 C = A @ B
    A: (M, K)
    B: (K, N)
    C: (M, N)
    """
    # 2D 程序网格
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    # 初始化累加器
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    # 沿 K 维度分块
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # 计算 A 的块指针
        a_offset_m = pid_m * BLOCK_SIZE_M
        a_offset_k = k * BLOCK_SIZE_K
        a_offsets = (
            a_offset_m + tl.arange(0, BLOCK_SIZE_M)[:, None]
        ) * stride_am + (
            a_offset_k + tl.arange(0, BLOCK_SIZE_K)[None, :]
        ) * stride_ak
        a_mask = (
            (a_offset_m + tl.arange(0, BLOCK_SIZE_M)[:, None] < M) &
            (a_offset_k + tl.arange(0, BLOCK_SIZE_K)[None, :] < K)
        )
        a = tl.load(a_ptr + a_offsets, mask=a_mask, other=0.0)
        
        # 计算 B 的块指针
        b_offset_k = k * BLOCK_SIZE_K
        b_offset_n = pid_n * BLOCK_SIZE_N
        b_offsets = (
            b_offset_k + tl.arange(0, BLOCK_SIZE_K)[:, None]
        ) * stride_bk + (
            b_offset_n + tl.arange(0, BLOCK_SIZE_N)[None, :]
        ) * stride_bn
        b_mask = (
            (b_offset_k + tl.arange(0, BLOCK_SIZE_K)[:, None] < K) &
            (b_offset_n + tl.arange(0, BLOCK_SIZE_N)[None, :] < N)
        )
        b = tl.load(b_ptr + b_offsets, mask=b_mask, other=0.0)
        
        # 累加部分乘积
        accumulator += tl.dot(a, b)
    
    # 写回结果
    c_offset_m = pid_m * BLOCK_SIZE_M
    c_offset_n = pid_n * BLOCK_SIZE_N
    c_offsets = (
        c_offset_m + tl.arange(0, BLOCK_SIZE_M)[:, None]
    ) * stride_cm + (
        c_offset_n + tl.arange(0, BLOCK_SIZE_N)[None, :]
    ) * stride_cn
    c_mask = (
        (c_offset_m + tl.arange(0, BLOCK_SIZE_M)[:, None] < M) &
        (c_offset_n + tl.arange(0, BLOCK_SIZE_N)[None, :] < N)
    )
    
    tl.store(c_ptr + c_offsets, accumulator, mask=c_mask)


def matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    M, K = a.shape
    K2, N = b.shape
    assert K == K2
    
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)
    
    BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K = 64, 64, 32
    grid = (triton.cdiv(M, BLOCK_SIZE_M), triton.cdiv(N, BLOCK_SIZE_N))
    
    matmul_kernel[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K,
    )
    
    return c


# ==============================================================================
# 测试代码
# ==============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("测试 1: 加权求和")
    print("=" * 80)
    x = torch.randn(128, 256, device='cuda')
    weight = torch.randn(256, device='cuda')
    
    # Triton 版本
    output_triton = weighted_sum(x, weight)
    
    # PyTorch 参考版本
    output_torch = (x * weight).sum(dim=1)
    
    print(f"输入形状: x={x.shape}, weight={weight.shape}")
    print(f"输出形状: {output_triton.shape}")
    print(f"最大误差: {(output_triton - output_torch).abs().max().item():.2e}")
    print(f"是否匹配: {torch.allclose(output_triton, output_torch, rtol=1e-4)}")
    
    print("\n" + "=" * 80)
    print("测试 2: 向量加法")
    print("=" * 80)
    a = torch.randn(10000, device='cuda')
    b = torch.randn(10000, device='cuda')
    
    c_triton = vector_add(a, b)
    c_torch = a + b
    
    print(f"最大误差: {(c_triton - c_torch).abs().max().item():.2e}")
    print(f"是否匹配: {torch.allclose(c_triton, c_torch)}")
    
    print("\n" + "=" * 80)
    print("测试 3: 矩阵乘法")
    print("=" * 80)
    A = torch.randn(128, 256, device='cuda')
    B = torch.randn(256, 512, device='cuda')
    
    C_triton = matmul(A, B)
    C_torch = A @ B
    
    print(f"输入形状: A={A.shape}, B={B.shape}")
    print(f"输出形状: {C_triton.shape}")
    print(f"最大误差: {(C_triton - C_torch).abs().max().item():.2e}")
    print(f"是否匹配: {torch.allclose(C_triton, C_torch, rtol=1e-3)}")


# ==============================================================================
# Triton 重要语法总结
# ==============================================================================

"""
1. 装饰器和编译时常量
   @triton.jit                    # 标记 Triton 内核
   BLOCK_SIZE: tl.constexpr       # 编译时常量（必须在编译时已知）

2. 程序索引
   tl.program_id(axis)            # 获取当前程序实例的索引
   
3. 创建张量和索引
   tl.arange(start, end)          # 创建索引向量 [start, start+1, ..., end-1]
   tl.zeros(shape, dtype)         # 创建零张量
   tl.full(shape, value, dtype)   # 创建常数张量
   
4. 内存操作
   tl.load(ptr, mask, other)      # 加载数据，mask 控制边界
   tl.store(ptr, value, mask)     # 存储数据
   
5. 块指针（高级抽象）
   tl.make_block_ptr(...)         # 创建块指针
   ptr.advance((offset1, offset2)) # 移动块指针
   
6. 数学操作
   +, -, *, /                     # 逐元素操作
   tl.sum(x, axis)                # 求和
   tl.max(x, axis)                # 最大值
   tl.dot(a, b)                   # 矩阵乘法
   tl.exp(x), tl.log(x)           # 指数和对数
   
7. 工具函数
   tl.cdiv(a, b)                  # 向上取整除法 ceil(a / b)
   
8. 广播
   x[None, :]                     # 在第0维添加维度
   x[:, None]                     # 在第1维添加维度
   
9. 启动内核
   grid = (num_blocks_x, num_blocks_y, num_blocks_z)
   kernel[grid](*args)
"""
