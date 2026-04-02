#!/usr/bin/env python3
"""
==========================================================================
从Goldberg论文到LLM训练：浮点数Hands-On实验
==========================================================================

本脚本通过7个可运行的实验，让你亲手体验浮点数在LLM训练中的核心概念。

每个实验对应Goldberg论文的一个关键章节，并桥接到实际的LLM训练场景。
"""

import struct
import numpy as np

# ========================================================================
# 工具函数
# ========================================================================

def float32_to_bits(f):
    """将float32转换为32位二进制字符串"""
    packed = struct.pack('>f', f)
    integer = struct.unpack('>I', packed)[0]
    return format(integer, '032b')

def bits_to_float32(bits_str):
    """将32位二进制字符串转换为float32"""
    integer = int(bits_str, 2)
    packed = struct.pack('>I', integer)
    return struct.unpack('>f', packed)[0]

def float64_to_bits(f):
    """将float64转换为64位二进制字符串"""
    packed = struct.pack('>d', f)
    integer = struct.unpack('>Q', packed)[0]
    return format(integer, '064b')

def float16_to_bits(f):
    """将float16转换为16位二进制字符串"""
    f16 = np.float16(f)
    integer = f16.view(np.uint16)
    return format(integer, '016b')

def bf16_to_bits(f):
    """将bfloat16模拟为16位二进制字符串（截断float32的低16位）"""
    bits32 = float32_to_bits(f)
    return bits32[:16]  # 取高16位

def print_section(title):
    """打印分节标题"""
    width = 72
    print("\n" + "=" * width)
    print(f"  {title}")
    print("=" * width)

def print_subsection(title):
    """打印子标题"""
    print(f"\n--- {title} ---")


# ========================================================================
# 实验1：位级别的浮点数拆解
# ========================================================================

def experiment_1_bit_dissection():
    """
    实验1：亲眼看到浮点数的每一个bit
    
    对应论文：Section 1.1 (Floating-Point Formats)
              Section 2.1 (Formats and Operations)
    
    目标：理解符号位、指数、尾数如何编码一个数字
    """
    print_section("实验1：位级别的浮点数拆解")
    
    test_values = [1.0, -1.0, 0.1, 0.5, 3.14, 65504.0, 1e-38, float('inf'), float('nan')]
    
    print("\n[Float32拆解]")
    print(f"{'值':>12s} | {'符号':>4s} | {'指数(8位)':>10s} | {'尾数(23位)':>25s} | {'实际指数':>8s}")
    print("-" * 80)
    
    for val in test_values:
        bits = float32_to_bits(val)
        sign = bits[0]
        exponent = bits[1:9]
        mantissa = bits[9:32]
        
        exp_val = int(exponent, 2)
        if exp_val == 0:
            actual_exp = "denorm"
        elif exp_val == 255:
            actual_exp = "特殊"
        else:
            actual_exp = str(exp_val - 127)
        
        # 对于NaN，显示nan而不是科学计数
        val_str = f"{val:>12.6g}" if not np.isnan(val) else "         NaN"
        print(f"{val_str} |    {sign} | {exponent} |  {mantissa} | {actual_exp:>8s}")
    
    # 重点讲解0.1的表示
    print_subsection("深入理解：为什么0.1不精确")
    val = 0.1
    bits = float32_to_bits(val)
    
    print(f"0.1 的float32二进制表示: {bits}")
    print(f"  符号: {bits[0]}")
    print(f"  指数: {bits[1:9]} = {int(bits[1:9], 2)} - 127 = {int(bits[1:9], 2) - 127}")
    print(f"  尾数: {bits[9:32]}")
    print(f"  实际值: 1.{bits[9:32]} × 2^({int(bits[1:9], 2) - 127})")
    
    # 验证：手动计算
    mantissa_val = 1.0
    for i, bit in enumerate(bits[9:32]):
        if bit == '1':
            mantissa_val += 2**(-(i+1))
    exp = int(bits[1:9], 2) - 127
    reconstructed = mantissa_val * (2**exp)
    
    print(f"\n  手动重建值: {reconstructed:.20f}")
    print(f"  真实0.1:    {0.1:.20f}")
    print(f"  差异:       {abs(reconstructed - 0.1):.2e}")
    print(f"\n  → 论文解释: 0.1在二进制中是无限循环小数，有限位数只能近似表示")
    
    # 展示隐藏位
    print_subsection("隐藏位(Hidden Bit)演示")
    print("Float32编码1.0:")
    bits_1 = float32_to_bits(1.0)
    print(f"  二进制: {bits_1}")
    print(f"  尾数全0: {bits_1[9:32]}")
    print(f"  但实际significand = 1.{bits_1[9:32]} = 1.0 （前导1被'隐藏'了）")
    print(f"  这个'白赚'的1位使24位存储获得了25位... 不对，是24位有效精度")
    print(f"  （23位存储 + 1位隐藏位 = 24位有效位）")


# ========================================================================
# 实验2：不同精度格式的对比
# ========================================================================

def experiment_2_precision_comparison():
    """
    实验2：Float32 / Float16 / BFloat16 的精度和范围对比
    
    对应论文：Section 1.2 (Relative Error and Ulps)
              Section 2.1.1 (Base), Section 2.1.2 (Precision)
    
    目标：直观感受不同格式的精度差异和范围限制
    """
    print_section("实验2：不同精度格式对比")
    
    # Machine epsilon
    print_subsection("Machine Epsilon（机器精度）")
    print("论文定义: ε = (β/2) × β^(-p)")
    print()
    
    # 实际计算machine epsilon
    for name, dtype, p in [("Float64", np.float64, 53), 
                            ("Float32", np.float32, 24),
                            ("Float16", np.float16, 11)]:
        eps = np.finfo(dtype).eps
        theoretical = 2.0**(1-p)
        print(f"  {name}: ε = {eps:.6e}  (理论值 2^(1-{p}) = {theoretical:.6e})")
    
    # BFloat16的epsilon（numpy没有bf16，手动计算）
    bf16_eps = 2.0**(1-8)  # p=8 for bf16
    print(f"  BFloat16: ε ≈ {bf16_eps:.6e}  (理论值 2^(1-8) = {bf16_eps:.6e})")
    
    # 范围对比
    print_subsection("数值范围对比")
    
    formats = {
        "Float64": {"max": np.finfo(np.float64).max, "tiny": np.finfo(np.float64).tiny},
        "Float32": {"max": np.finfo(np.float32).max, "tiny": np.finfo(np.float32).tiny},
        "Float16": {"max": np.finfo(np.float16).max, "tiny": np.finfo(np.float16).tiny},
        "BFloat16": {"max": 3.39e38, "tiny": 1.18e-38},  # 和Float32相同的指数范围
    }
    
    print(f"{'格式':>10s} | {'最大值':>15s} | {'最小正归一化数':>15s} | {'指数范围':>15s}")
    print("-" * 65)
    for name, vals in formats.items():
        exp_range = {
            "Float64": "[-1022, 1023]",
            "Float32": "[-126, 127]", 
            "Float16": "[-14, 15]",
            "BFloat16": "[-126, 127]"
        }
        print(f"{name:>10s} | {vals['max']:>15.3e} | {vals['tiny']:>15.3e} | {exp_range[name]:>15s}")
    
    # 精度实验：1 + small_number
    print_subsection("精度实验: 1.0 + small_number")
    print("测试哪些小数能被不同格式'看到'：\n")
    
    small_numbers = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8]
    
    print(f"{'small_number':>12s} | {'FP32结果':>14s} | {'FP16结果':>14s} | {'BF16结果':>14s}")
    print("-" * 65)
    
    for sn in small_numbers:
        fp32_result = np.float32(1.0) + np.float32(sn)
        fp16_result = np.float16(1.0) + np.float16(sn)
        # BF16模拟：先在FP32中计算，然后截断到BF16精度
        bf16_result_full = np.float32(1.0) + np.float32(sn)
        # 模拟BF16截断：将float32的低16位尾数置零
        bf16_int = struct.unpack('>I', struct.pack('>f', bf16_result_full))[0]
        bf16_int = bf16_int & 0xFFFF0000  # 截断低16位
        bf16_result = struct.unpack('>f', struct.pack('>I', bf16_int))[0]
        
        fp32_ok = "✓" if fp32_result != 1.0 else "✗(=1.0)"
        fp16_ok = "✓" if fp16_result != 1.0 else "✗(=1.0)"
        bf16_ok = "✓" if bf16_result != 1.0 else "✗(=1.0)"
        
        print(f"  {sn:>10.0e} | {fp32_ok:>14s} | {fp16_ok:>14s} | {bf16_ok:>14s}")
    
    print("\n  → BFloat16的精度比Float16更差（ε更大），但范围和Float32一样大")
    print("  → 这就是为什么BFloat16不需要Loss Scaling：梯度不会溢出/下溢")
    
    # 溢出实验
    print_subsection("溢出实验")
    
    print("测试较大数值的平方：")
    test_vals = [100.0, 256.0, 500.0, 1000.0]
    for v in test_vals:
        fp16_v = np.float16(v)
        fp16_sq = np.float16(v) * np.float16(v)
        fp32_sq = np.float32(v) * np.float32(v)
        
        status = "溢出!" if np.isinf(fp16_sq) else "正常"
        print(f"  {v}² : FP32={fp32_sq:.0f}, FP16={fp16_sq} ({status})")
    
    print(f"\n  → Float16最大值只有{np.finfo(np.float16).max}，很容易溢出")
    print("  → BFloat16最大值约3.4×10^38，和Float32一样，溢出极少发生")


# ========================================================================
# 实验3：灾难性抵消
# ========================================================================

def experiment_3_catastrophic_cancellation():
    """
    实验3：灾难性抵消的现场演示
    
    对应论文：Section 1.4 (Cancellation)
    
    目标：理解为什么两个接近的数相减会丢失精度，以及如何避免
    """
    print_section("实验3：灾难性抵消（Catastrophic Cancellation）")
    
    # 论文中的二次方程例子
    print_subsection("例1：二次方程（论文Section 1.4）")
    
    # b² ≈ 4ac 的情况
    a, b, c = 1.0, 1e8, 1.0
    
    print(f"  求解 ax² + bx + c = 0，其中 a={a}, b={b:.0e}, c={c}")
    
    # 直接用公式
    discriminant = b**2 - 4*a*c
    r1_naive = (-b + np.sqrt(discriminant)) / (2*a)
    r2_naive = (-b - np.sqrt(discriminant)) / (2*a)
    
    # 用改进公式（论文eq. 5）
    sqrt_disc = np.sqrt(discriminant)
    if b > 0:
        r2_improved = (-b - sqrt_disc) / (2*a)
        r1_improved = (2*c) / (-b - sqrt_disc)
    else:
        r1_improved = (-b + sqrt_disc) / (2*a)
        r2_improved = (2*c) / (-b + sqrt_disc)
    
    # 用numpy的高精度求解作为参考
    r1_exact = -1e-8  # 精确值
    r2_exact = -1e8   # 精确值
    
    print(f"\n  朴素公式:   r₁ = {r1_naive:.15e}")
    print(f"  改进公式:   r₁ = {r1_improved:.15e}")
    print(f"  精确值:     r₁ = {r1_exact:.15e}")
    print(f"\n  朴素公式相对误差:  {abs(r1_naive - r1_exact)/abs(r1_exact):.2e}")
    print(f"  改进公式相对误差:  {abs(r1_improved - r1_exact)/abs(r1_exact):.2e}")
    print(f"\n  → 朴素公式中 -b + √(b²-4ac) ≈ -10⁸ + 10⁸ 发生灾难性抵消")
    print(f"  → 改进公式通过代数恒等变换避免了这个减法")
    
    # 论文中的三角形面积例子
    print_subsection("例2：三角形面积（论文Section 1.4, Theorem 3）")
    
    # 一个非常"扁"的三角形
    a_side = np.float32(9.0)
    b_side = np.float32(4.53)
    c_side = np.float32(4.53)
    
    # 标准Heron公式
    s = np.float32((a_side + b_side + c_side) / 2)
    area_heron = np.float32(np.sqrt(s * (s - a_side) * (s - b_side) * (s - c_side)))
    
    # 论文改进公式 (eq. 7)，使用float32
    sides = sorted([float(a_side), float(b_side), float(c_side)], reverse=True)
    a_s, b_s, c_s = [np.float32(x) for x in sides]
    
    term1 = np.float32(a_s + (b_s + c_s))
    term2 = np.float32(c_s - (a_s - b_s))
    term3 = np.float32(c_s + (a_s - b_s))
    term4 = np.float32(a_s + (b_s - c_s))
    
    area_improved = np.float32(np.sqrt(term1 * term2 * term3 * term4) / 4)
    
    # 高精度参考值
    s_exact = (9.0 + 4.53 + 4.53) / 2
    area_exact = np.sqrt(s_exact * (s_exact - 9.0) * (s_exact - 4.53) * (s_exact - 4.53))
    
    print(f"  扁三角形: a={a_side}, b={b_side}, c={c_side}")
    print(f"  Heron公式面积:  {area_heron:.6f}")
    print(f"  改进公式面积:   {area_improved:.6f}")
    print(f"  高精度参考:     {area_exact:.6f}")
    print(f"\n  Heron误差: {abs(float(area_heron) - area_exact)/area_exact:.2e}")
    print(f"  改进误差:   {abs(float(area_improved) - area_exact)/area_exact:.2e}")
    print(f"\n  → 论文Theorem 3证明改进公式的误差最多11ε")
    
    # LLM训练中的实际案例：方差计算
    print_subsection("例3：LayerNorm中的方差计算（LLM实际问题）")
    
    # 模拟一个均值很大、方差很小的激活值分布
    np.random.seed(42)
    x = np.float32(1000.0 + np.random.normal(0, 0.01, 1000).astype(np.float32))
    
    # 不好的方式：E[x²] - (E[x])²
    var_bad = np.float32(np.mean(x**2) - np.mean(x)**2)
    
    # 好的方式：E[(x - mean)²]
    mean_x = np.mean(x)
    var_good = np.float32(np.mean((x - mean_x)**2))
    
    # 高精度参考
    var_exact = np.var(x.astype(np.float64))
    
    print(f"  数据: 1000个数，均值≈1000，标准差≈0.01")
    print(f"  E[x²]-(E[x])²:    {var_bad:.10f}")
    print(f"  E[(x-mean)²]:     {var_good:.10f}")
    print(f"  Float64参考:       {var_exact:.10f}")
    
    if var_bad < 0:
        print(f"\n  ⚠ E[x²]-(E[x])²竟然得到了负数！方差不可能为负！")
        print(f"  这就是灾难性抵消的后果：E[x²]≈10^6, (E[x])²≈10^6，相减丢失精度")
    else:
        rel_err_bad = abs(var_bad - var_exact) / var_exact if var_exact > 0 else float('inf')
        rel_err_good = abs(var_good - var_exact) / var_exact if var_exact > 0 else float('inf')
        print(f"\n  不好方式的相对误差: {rel_err_bad:.2e}")
        print(f"  好方式的相对误差:   {rel_err_good:.2e}")
    
    print(f"\n  → 实际LLM实现中，LayerNorm总是使用 E[(x-mean)²] 的形式")


# ========================================================================
# 实验4：Kahan求和 vs 朴素求和
# ========================================================================

def experiment_4_kahan_summation():
    """
    实验4：Kahan求和算法
    
    对应论文：Section 4.3 (Errors in Summation), Theorem 8
    
    目标：理解为什么长求和会有误差累积，以及如何修复
    """
    print_section("实验4：Kahan求和 vs 朴素求和")
    
    print("论文Theorem 8: Kahan求和把误差从 O(nε) 降低到 O(ε)")
    
    def naive_sum_fp32(arr):
        """朴素求和，用float32"""
        s = np.float32(0.0)
        for x in arr:
            s = np.float32(s + np.float32(x))
        return s
    
    def kahan_sum_fp32(arr):
        """Kahan求和，用float32"""
        s = np.float32(0.0)
        c = np.float32(0.0)  # 补偿值
        for x in arr:
            x = np.float32(x)
            y = np.float32(x - c)        # 减去上次的补偿
            t = np.float32(s + y)         # 加到总和
            c = np.float32(np.float32(t - s) - y)  # 计算丢失的低位
            s = t
        return s
    
    def naive_sum_fp16(arr):
        """朴素求和，用float16"""
        s = np.float16(0.0)
        for x in arr:
            s = np.float16(s + np.float16(x))
        return s
    
    # 测试1：大量小数求和
    print_subsection("测试1：10000个0.1的和（应该=1000）")
    
    n = 10000
    arr = [0.1] * n
    exact = n * 0.1
    
    naive_32 = naive_sum_fp32(arr)
    kahan_32 = kahan_sum_fp32(arr)
    numpy_32 = np.sum(np.float32(arr))
    naive_16 = naive_sum_fp16(arr)
    
    print(f"  精确值:        {exact}")
    print(f"  朴素求和FP32:  {naive_32}  (误差: {abs(float(naive_32)-exact):.6f})")
    print(f"  Kahan求和FP32: {kahan_32}  (误差: {abs(float(kahan_32)-exact):.6f})")
    print(f"  NumPy求和FP32: {numpy_32}  (误差: {abs(float(numpy_32)-exact):.6f})")
    print(f"  朴素求和FP16:  {naive_16}  (误差: {abs(float(naive_16)-exact):.6f})")
    
    # 测试2：模拟梯度累加场景
    print_subsection("测试2：模拟梯度累加（大数+小梯度）")
    
    np.random.seed(42)
    # 模拟：权重≈10，梯度≈0.001
    weight = np.float32(10.0)
    gradients = np.random.normal(0.001, 0.0005, 5000).astype(np.float32)
    
    # 在FP32中朴素累加
    sum_naive_32 = np.float32(0.0)
    for g in gradients:
        sum_naive_32 = np.float32(sum_naive_32 + g)
    
    # 在FP32中Kahan累加
    sum_kahan_32 = kahan_sum_fp32(gradients)
    
    # 在FP16中朴素累加
    sum_naive_16 = np.float16(0.0)
    for g in gradients:
        sum_naive_16 = np.float16(sum_naive_16 + np.float16(g))
    
    # 高精度参考
    sum_exact = np.sum(gradients.astype(np.float64))
    
    print(f"  5000个梯度之和:")
    print(f"  Float64参考:   {sum_exact:.10f}")
    print(f"  FP32朴素:      {float(sum_naive_32):.10f}  (误差: {abs(float(sum_naive_32)-sum_exact):.2e})")
    print(f"  FP32 Kahan:    {float(sum_kahan_32):.10f}  (误差: {abs(float(sum_kahan_32)-sum_exact):.2e})")
    print(f"  FP16朴素:      {float(sum_naive_16):.10f}  (误差: {abs(float(sum_naive_16)-sum_exact):.2e})")
    
    print(f"\n  → 在实际LLM训练中，梯度累加在FP32中进行")
    print(f"  → GPU Tensor Core: 乘法用FP16/BF16，但累加用FP32内部寄存器")


# ========================================================================
# 实验5：INT8/INT4量化模拟
# ========================================================================

def experiment_5_quantization():
    """
    实验5：整数量化模拟
    
    对应论文：对浮点数均匀/非均匀间距的讨论 (Section 1.1, Figure 1)
    
    目标：理解量化的精度损失和不同粒度策略
    """
    print_section("实验5：INT8/INT4量化模拟")
    
    def quantize_symmetric(tensor, num_bits):
        """对称量化"""
        qmax = 2**(num_bits - 1) - 1
        qmin = -qmax
        
        abs_max = np.max(np.abs(tensor))
        scale = abs_max / qmax if abs_max > 0 else 1.0
        
        quantized = np.clip(np.round(tensor / scale), qmin, qmax).astype(np.int8 if num_bits == 8 else np.int32)
        dequantized = quantized.astype(np.float32) * scale
        
        return quantized, dequantized, scale
    
    def quantize_per_group(tensor, num_bits, group_size):
        """Per-group量化"""
        qmax = 2**(num_bits - 1) - 1
        qmin = -qmax
        
        # Reshape为groups
        n = len(tensor)
        n_groups = (n + group_size - 1) // group_size
        padded = np.zeros(n_groups * group_size, dtype=np.float32)
        padded[:n] = tensor
        groups = padded.reshape(n_groups, group_size)
        
        # 每组独立量化
        dequantized = np.zeros_like(padded)
        for i in range(n_groups):
            group = groups[i]
            abs_max = np.max(np.abs(group))
            scale = abs_max / qmax if abs_max > 0 else 1.0
            q = np.clip(np.round(group / scale), qmin, qmax)
            dequantized[i*group_size:(i+1)*group_size] = q * scale
        
        return dequantized[:n]
    
    # 模拟一个典型的LLM权重分布（近似正态分布）
    np.random.seed(42)
    weights = np.random.normal(0, 0.02, 4096).astype(np.float32)  # 典型LLM权重
    
    print_subsection("权重分布统计")
    print(f"  权重数量: {len(weights)}")
    print(f"  均值: {np.mean(weights):.6f}")
    print(f"  标准差: {np.std(weights):.6f}")
    print(f"  范围: [{np.min(weights):.6f}, {np.max(weights):.6f}]")
    
    print_subsection("不同量化精度的误差")
    
    for bits in [8, 4]:
        _, dequant, scale = quantize_symmetric(weights, bits)
        mse = np.mean((weights - dequant)**2)
        max_err = np.max(np.abs(weights - dequant))
        rel_err = np.sqrt(mse) / np.std(weights)
        
        print(f"\n  INT{bits} (Per-tensor对称量化):")
        print(f"    Scale: {scale:.8f}")
        print(f"    MSE:   {mse:.2e}")
        print(f"    最大误差: {max_err:.2e}")
        print(f"    相对RMSE: {rel_err:.4f} ({rel_err*100:.2f}%)")
        print(f"    可表示的值: {2**bits}个 (范围: [-{2**(bits-1)-1}, {2**(bits-1)-1}])")
    
    print_subsection("Per-group量化的改进 (INT4)")
    
    for group_size in [4096, 128, 32]:
        dequant = quantize_per_group(weights, 4, group_size)
        mse = np.mean((weights - dequant)**2)
        rel_err = np.sqrt(mse) / np.std(weights)
        
        label = "Per-tensor" if group_size >= len(weights) else f"g={group_size}"
        print(f"    INT4 {label:>12s}: 相对RMSE = {rel_err:.4f} ({rel_err*100:.2f}%)")
    
    print(f"\n  → Per-group量化（g=128）是当前LLM推理量化的主流方案")
    print(f"  → 更小的group_size → 更好的精度，但更多的scale需要存储")
    
    # 量化对矩阵乘法的影响
    print_subsection("量化对矩阵乘法的影响")
    
    # 模拟 y = Wx
    W = np.random.normal(0, 0.02, (256, 256)).astype(np.float32)
    x = np.random.normal(0, 1.0, 256).astype(np.float32)
    
    y_exact = W @ x
    
    # INT8量化W
    W_flat = W.flatten()
    _, W_dequant_flat, _ = quantize_symmetric(W_flat, 8)
    W_int8 = W_dequant_flat.reshape(W.shape)
    y_int8 = W_int8 @ x
    
    # INT4量化W
    _, W_dequant_flat4, _ = quantize_symmetric(W_flat, 4)
    W_int4 = W_dequant_flat4.reshape(W.shape)
    y_int4 = W_int4 @ x
    
    cosine_int8 = np.dot(y_exact, y_int8) / (np.linalg.norm(y_exact) * np.linalg.norm(y_int8))
    cosine_int4 = np.dot(y_exact, y_int4) / (np.linalg.norm(y_exact) * np.linalg.norm(y_int4))
    
    rel_err_int8 = np.sqrt(np.mean((y_exact - y_int8)**2)) / np.sqrt(np.mean(y_exact**2))
    rel_err_int4 = np.sqrt(np.mean((y_exact - y_int4)**2)) / np.sqrt(np.mean(y_exact**2))
    
    print(f"  矩阵乘法 y = Wx (W: 256×256, x: 256)")
    print(f"  INT8: 余弦相似度={cosine_int8:.6f}, 相对误差={rel_err_int8:.4f}")
    print(f"  INT4: 余弦相似度={cosine_int4:.6f}, 相对误差={rel_err_int4:.4f}")
    print(f"\n  → 单次矩阵乘法的误差可以接受，但多层传播后会累积")


# ========================================================================
# 实验6：混合精度权重更新
# ========================================================================

def experiment_6_mixed_precision_update():
    """
    实验6：为什么FP32 master copy必不可少
    
    对应论文：Section 1.2 (machine epsilon导致小量被吃掉)
    
    目标：看到在低精度中权重更新如何"消失"
    """
    print_section("实验6：混合精度权重更新模拟")
    
    print("模拟1000步梯度下降，观察不同精度策略的差异")
    
    np.random.seed(42)
    
    # 初始权重
    w_init = 5.0
    lr = 1e-3
    
    # 模拟梯度（朝着减小权重的方向）
    gradients = np.random.normal(0.1, 0.05, 1000)  # 平均梯度0.1
    
    # 方案1：全FP32
    w_fp32 = np.float32(w_init)
    history_fp32 = [float(w_fp32)]
    for g in gradients:
        w_fp32 = np.float32(w_fp32 - np.float32(lr * g))
        history_fp32.append(float(w_fp32))
    
    # 方案2：全FP16（错误做法）
    w_fp16 = np.float16(w_init)
    history_fp16 = [float(w_fp16)]
    for g in gradients:
        update = np.float16(np.float16(lr) * np.float16(g))
        w_fp16 = np.float16(w_fp16 - update)
        history_fp16.append(float(w_fp16))
    
    # 方案3：全BF16（错误做法，模拟）
    def to_bf16(val):
        """模拟BFloat16：截断float32的低16位尾数"""
        f32 = np.float32(val)
        int_val = struct.unpack('>I', struct.pack('>f', f32))[0]
        int_val = int_val & 0xFFFF0000
        return struct.unpack('>f', struct.pack('>I', int_val))[0]
    
    w_bf16 = to_bf16(w_init)
    history_bf16 = [w_bf16]
    for g in gradients:
        update = to_bf16(lr * g)
        w_bf16 = to_bf16(w_bf16 - update)
        history_bf16.append(w_bf16)
    
    # 方案4：混合精度（正确做法）：FP32 master + BF16 forward
    w_master = np.float32(w_init)  # FP32 master copy
    history_mixed = [float(w_master)]
    for g in gradients:
        g_fp32 = np.float32(g)  # 梯度转为FP32
        w_master = np.float32(w_master - np.float32(lr * g_fp32))  # FP32中更新
        history_mixed.append(float(w_master))
    
    # 高精度参考
    w_exact = np.float64(w_init)
    for g in gradients:
        w_exact = w_exact - np.float64(lr) * np.float64(g)
    
    print_subsection("1000步后的结果")
    
    print(f"  Float64参考:         {w_exact:.10f}")
    print(f"  全FP32:              {history_fp32[-1]:.10f}  (误差: {abs(history_fp32[-1]-w_exact):.2e})")
    print(f"  全FP16 (错误做法):   {history_fp16[-1]:.10f}  (误差: {abs(history_fp16[-1]-w_exact):.2e})")
    print(f"  全BF16 (错误做法):   {history_bf16[-1]:.10f}  (误差: {abs(history_bf16[-1]-w_exact):.2e})")
    print(f"  混合精度 (正确):     {history_mixed[-1]:.10f}  (误差: {abs(history_mixed[-1]-w_exact):.2e})")
    
    # 分析FP16的问题
    print_subsection("分析：为什么FP16更新'消失'了")
    
    typical_update = lr * 0.1  # = 0.0001
    w_val = 5.0
    
    fp16_eps = np.finfo(np.float16).eps
    fp16_ulp_at_5 = np.float16(5.0) * fp16_eps
    
    bf16_eps = 2.0**(1-8)
    bf16_ulp_at_5 = 5.0 * bf16_eps
    
    fp32_eps = np.finfo(np.float32).eps
    fp32_ulp_at_5 = np.float32(5.0) * fp32_eps
    
    print(f"  典型更新量: lr × gradient = {lr} × 0.1 = {typical_update}")
    print(f"\n  在w=5.0处的ULP（最小可区分变化量）:")
    print(f"    FP16:  {fp16_ulp_at_5:.2e}  → 更新量{typical_update}/{fp16_ulp_at_5:.2e} = {typical_update/fp16_ulp_at_5:.1f} ulps")
    print(f"    BF16:  {bf16_ulp_at_5:.2e}  → 更新量{typical_update}/{bf16_ulp_at_5:.2e} = {typical_update/bf16_ulp_at_5:.1f} ulps")
    print(f"    FP32:  {fp32_ulp_at_5:.2e}  → 更新量{typical_update}/{fp32_ulp_at_5:.2e} = {typical_update/fp32_ulp_at_5:.1f} ulps")
    
    print(f"\n  → FP16中，更新量不到1个ulp，可能被完全舍入掉!")
    print(f"  → BF16中，更新量约5个ulp，也容易有较大舍入误差")
    print(f"  → FP32中，更新量约168个ulp，精度充足")
    print(f"\n  → 结论：权重更新必须在FP32中进行（master copy）")
    
    # 展示训练进度对比
    print_subsection("训练进度对比（每200步）")
    print(f"  {'步数':>6s} | {'FP64参考':>12s} | {'FP32':>12s} | {'FP16':>12s} | {'BF16':>12s} | {'混合':>12s}")
    print("  " + "-" * 75)
    
    # 重新计算FP64的中间过程
    w_exact_hist = [float(w_init)]
    w_e = np.float64(w_init)
    for g in gradients:
        w_e = w_e - np.float64(lr) * np.float64(g)
        w_exact_hist.append(float(w_e))
    
    for step in range(0, 1001, 200):
        print(f"  {step:>6d} | {w_exact_hist[step]:>12.6f} | {history_fp32[step]:>12.6f} | "
              f"{history_fp16[step]:>12.6f} | {history_bf16[step]:>12.6f} | {history_mixed[step]:>12.6f}")


# ========================================================================
# 实验7：Loss Scaling
# ========================================================================

def experiment_7_loss_scaling():
    """
    实验7：Loss Scaling——为什么Float16训练需要它
    
    对应论文：Section 2.2.4 (Denormalized Numbers, gradual underflow)
              Section 2.3 (Exceptions)
    
    目标：理解下溢问题和Loss Scaling的解决方案
    """
    print_section("实验7：Loss Scaling实验")
    
    print("当梯度很小时，Float16会下溢为0。Loss Scaling通过放大来避免。")
    
    # 模拟不同大小的梯度在不同格式中的表示
    print_subsection("不同大小的梯度在各格式中的表示")
    
    gradient_magnitudes = [1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8]
    
    print(f"  {'梯度大小':>10s} | {'FP32':>14s} | {'FP16':>14s} | {'BF16(模拟)':>14s}")
    print("  " + "-" * 60)
    
    for g in gradient_magnitudes:
        fp32_g = np.float32(g)
        fp16_g = np.float16(g)
        
        # BF16模拟
        bf16_int = struct.unpack('>I', struct.pack('>f', np.float32(g)))[0]
        bf16_int = bf16_int & 0xFFFF0000
        bf16_g = struct.unpack('>f', struct.pack('>I', bf16_int))[0]
        
        fp16_status = f"{float(fp16_g):.2e}" if fp16_g != 0 else "0 (下溢!)"
        bf16_status = f"{bf16_g:.2e}" if bf16_g != 0 else "0 (下溢!)"
        
        print(f"  {g:>10.0e} | {float(fp32_g):>14.2e} | {fp16_status:>14s} | {bf16_status:>14s}")
    
    print(f"\n  FP16最小正归一化数: {np.finfo(np.float16).tiny:.2e}")
    print(f"  BF16最小正归一化数: ≈ {np.finfo(np.float32).tiny:.2e} (和FP32一样)")
    
    # Loss Scaling演示
    print_subsection("Loss Scaling演示")
    
    original_grad = 1e-6
    scale_factor = 1024.0  # 2^10
    
    print(f"  原始梯度: {original_grad:.2e}")
    print(f"  FP16表示: {float(np.float16(original_grad)):.2e}  → {'下溢!' if np.float16(original_grad) == 0 else '正常'}")
    
    scaled_grad = original_grad * scale_factor
    print(f"\n  Loss Scale = {scale_factor}")
    print(f"  放大后梯度: {scaled_grad:.2e}")
    print(f"  FP16表示: {float(np.float16(scaled_grad)):.2e}  → {'下溢!' if np.float16(scaled_grad) == 0 else '正常 ✓'}")
    
    # 在FP32中缩回
    recovered = np.float32(np.float16(scaled_grad)) / np.float32(scale_factor)
    print(f"  缩回后(FP32): {float(recovered):.2e}")
    print(f"  误差: {abs(float(recovered) - original_grad)/original_grad:.2e}")
    
    # 动态Loss Scaling模拟
    print_subsection("动态Loss Scaling模拟")
    
    np.random.seed(42)
    scale = np.float32(2**15)  # 初始scale
    n_stable = 0
    growth_interval = 200  # 连续200步没问题就翻倍
    
    print(f"  初始scale: {scale}")
    print(f"  模拟1000步训练...\n")
    
    overflow_count = 0
    scale_changes = []
    
    for step in range(1000):
        # 模拟梯度（偶尔有大梯度）
        if np.random.random() < 0.02:  # 2%概率出现大梯度
            grad = np.float32(np.random.uniform(100, 1000))
        else:
            grad = np.float32(np.random.normal(0.01, 0.005))
        
        # 放大
        scaled_grad = np.float16(grad * scale)
        
        # 检查溢出
        if np.isinf(scaled_grad) or np.isnan(scaled_grad):
            overflow_count += 1
            scale = np.float32(scale / 2)  # scale减半
            n_stable = 0
            scale_changes.append((step, float(scale), "↓ overflow"))
            continue  # 跳过此步
        
        # 正常更新
        n_stable += 1
        if n_stable >= growth_interval:
            scale = np.float32(scale * 2)  # scale翻倍
            n_stable = 0
            scale_changes.append((step, float(scale), "↑ growth"))
    
    print(f"  溢出次数: {overflow_count}")
    print(f"  Scale变化记录（最后10次）:")
    for step, s, reason in scale_changes[-10:]:
        print(f"    Step {step:>4d}: scale={s:>12.0f}  ({reason})")
    print(f"  最终scale: {scale}")
    
    print(f"\n  → BFloat16不需要这个机制，因为指数范围足够大")
    print(f"  → 这也是BFloat16越来越受欢迎的主要原因之一")


# ========================================================================
# 主函数
# ========================================================================

def main():
    print("╔" + "═" * 70 + "╗")
    print("║  从Goldberg论文到LLM训练：浮点数 Hands-On 实验                    ║")
    print("║  共7个实验，每个对应论文的核心章节                                 ║")
    print("╚" + "═" * 70 + "╝")
    
    experiment_1_bit_dissection()
    experiment_2_precision_comparison()
    experiment_3_catastrophic_cancellation()
    experiment_4_kahan_summation()
    experiment_5_quantization()
    experiment_6_mixed_precision_update()
    experiment_7_loss_scaling()
    
    print("\n" + "=" * 72)
    print("  全部实验完成！")
    print("=" * 72)
    print("""
总结——从论文到实践的核心映射：

  论文概念                    LLM训练中的体现
  ─────────────────────────────────────────────────
  Machine epsilon (ε)      → 决定权重更新的最小有效量
  Guard digit / 2ε bound   → Tensor Core中FP32累加器
  灾难性抵消               → LayerNorm、Softmax的数值稳定实现
  Kahan求和 / 误差累积     → 梯度累加必须用FP32
  Overflow / Underflow     → FP16需要Loss Scaling
  指数范围 vs 精度         → BFloat16：牺牲精度保指数范围
  Denormalized numbers     → 保证 x≠y → x-y≠0 的正确性
""")


if __name__ == "__main__":
    main()
