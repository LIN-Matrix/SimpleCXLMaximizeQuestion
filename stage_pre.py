import prettytable as pt
import pulp

"本代码中所有的时间参数的单位都是秒(s),所有的数据大小参数的单位都是字节(B),因为1个Float占4B,所以计算公式中出现较多的常数4"
# 硬件参数
GPU_MEMORY = 128 * 1024**3  # GPU的显存的大小，128 GB 「1个32位Float占4B」
FLOPS_GPU = 312 * 10**12  # GPU计算浮点数乘法速度，312 TFLOPS 「Tera Floating Point Operations Per Second」
FLOPS_CXL = 4 * 10**12  # CXL计算浮点数乘法速度，4 TFLOPS
BAND_WIDTH = 128 * 1024**3  # GPU和CXL之间的传输速度，128 GB/s
LATENCY = 1 * 10**-6  # GPU和CXL之间的传输延迟，1微秒

# 以参数表格形式输出硬件参数
table = pt.PrettyTable()
table.field_names = ["参数", "值", "单位"]
table.add_row(["GPU_MEMORY", f"{GPU_MEMORY / 1024**3:.2f}", "GB"])
table.add_row(["FLOPS_GPU", f"{FLOPS_GPU / 10**12:.2f}", "TFLOPS"])
table.add_row(["FLOPS_CXL", f"{FLOPS_CXL / 10**12:.2f}", "TFLOPS"])
table.add_row(["BAND_WIDTH", f"{BAND_WIDTH / 1024**3:.2f}", "GB/s"])
table.add_row(["LATENCY", f"{LATENCY / 10**-6:.2f}", "us"])
print(table)

# 定义变量
Model_dim = 512    # 模型Pretrained Weights（d*d的矩阵）的维度d，设置为512
Lora_dim = 4       # Lora微调的（A和B都是d*r的矩阵）的规模r，设置为4，r<<d
# Batch_size = 0  # Batch size 需要保证GPU在计算中总存储容量不超过GPU_MEMORY，同时尽可能大，来保证高的Throughput，因为Throughput = Batch_size / Time

# 定义线性规划问题
prob = pulp.LpProblem("Memory_Optimization", pulp.LpMaximize)
# 定义正整数变量 batch1, batch2, batch3
b1 = pulp.LpVariable('b1', lowBound=0, upBound=1, cat='Integer')
b2 = pulp.LpVariable('b2', lowBound=0, upBound=1, cat='Integer')
b3 = pulp.LpVariable('b3', lowBound=0, upBound=1, cat='Integer')
# 约束条件：最开始对于GPU的内存使用不超过GPU_MEMORY
prob += 4 * (b1 + b2 + b3) * Model_dim <= GPU_MEMORY # b1、b2、b3的内存使用


"第一阶段(Stage1),计算Y = XA,GPU把 batch2 和 batch3 的部分的矩阵卸载到CXL中计算,在GPU中计算 batch1 的部分,CXL计算并将 batch2 结果传回GPU,传回时刻对齐,对结果进行整合"
# GPU第一阶段耗时为：计算Y = XA
T_GPU_S1 = 2 * b1 * Model_dim * Lora_dim / FLOPS_GPU
# CXL第一阶段耗时为：指令延迟 + 传输b2+b3数据 + 计算Y = XA + 传输b2数据
T_CXL_S1 = LATENCY + 4 * (b2 + b3) * Model_dim / BAND_WIDTH + 2 * (b2 + b3) * Model_dim * Lora_dim / FLOPS_CXL  + 4 * b2 * Lora_dim / BAND_WIDTH
# 约束条件：CXL第一阶段的时间不超过GPU第一阶段的时间，能够赶上第一班“车”
prob += T_CXL_S1 <= T_GPU_S1
# 约束条件：第一阶段计算过程中和计算后拼接b1和b2，对于GPU的内存使用不超过GPU_MEMORY
prob += 4 * b1 * Model_dim + 4 * Model_dim * Lora_dim + 4 * b1 * Lora_dim <= GPU_MEMORY # X(b1)、A、Y(b1)的内存使用
prob += 4 * (b1 + b2) * Lora_dim <= GPU_MEMORY # b1、b2的内存使用
# 第一阶段总耗时
print("T_GPU_S1:", T_GPU_S1, "T_CXL_S1:", T_CXL_S1)
# 第一阶段总内存使用
print("M_GPU_S1-1:", 4 * b1 * Model_dim + 4 * Model_dim * Lora_dim + 4 * b1 * Lora_dim, "M_GPU_S1-2:", 4 * (b1 + b2) * Lora_dim)


"第二阶段(Stage2),计算Z = YB,在GPU中计算 batch1 和 batch2 的部分,CXL计算并将 batch3 结果传回GPU,传回时刻对齐,对结果进行整合"
# GPU第二阶段耗时为：计算Z = YB
T_GPU_S2 = 2 * (b1 + b2) * Lora_dim * Model_dim / FLOPS_GPU
# CXL第二阶段耗时为：计算Z = YB + 传输b3数据 - 传输b2数据 TODO: 传输b3是否需要指令，是否需要考虑LATENCY？
T_CXL_S2 = 2 * b3 * Lora_dim * Model_dim / FLOPS_CXL + 4 * b3 * Model_dim / BAND_WIDTH - 4 * b2 * Lora_dim / BAND_WIDTH
# 约束条件：CXL第二阶段的时间不超过GPU第二阶段的时间，能够赶上第二班“车”
prob += T_CXL_S2 <= T_GPU_S2
# 约束条件：第二阶段计算过程中和计算后拼接b1b2和b3，对于GPU的内存使用不超过GPU_MEMORY
prob += 4 * (b1 + b2) * Lora_dim + 4 * Lora_dim * Model_dim + 4 * (b1 + b2) * Model_dim <= GPU_MEMORY # Y(b1b2)、B、Z(b1b2)的内存使用
prob += 4 * (b1 + b2 + b3) * Model_dim <= GPU_MEMORY # b1、b2、b3的内存使用
# 第二阶段总耗时
print("T_GPU_S2:", T_GPU_S2, "T_CXL_S2:", T_CXL_S2)
# 第二阶段总内存使用
print("M_GPU_S2-1:", 4 * (b1 + b2) * Lora_dim + 4 * Lora_dim * Model_dim + 4 * (b1 + b2) * Model_dim, "M_GPU_S2-2:", 4 * (b1 + b2 + b3) * Model_dim)


# 目标函数：最大化吞吐率
prob += (b1 + b2 + b3) / (T_GPU_S1+ T_GPU_S2)
# 输出线性规划问题
print(prob)
# 求解问题
prob.solve()
# 输出结果
print("Status:", pulp.LpStatus[prob.status])
print("b1:", pulp.value(b1))
print("b2:", pulp.value(b2))
print("b3:", pulp.value(b3))
print("Throughput:", pulp.value(prob.objective))
