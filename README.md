<img width="577" alt="image" src="https://github.com/user-attachments/assets/789bc76f-3688-4760-8a2d-118f31a8a615">
<img width="577" alt="image" src="https://github.com/user-attachments/assets/933302b5-17e9-48f2-b627-593e9528e23d">
<img width="677" alt="image" src="https://github.com/user-attachments/assets/d121bb3d-d917-456b-8f7d-48dbf66f3ea3">


### Mathematical Formulation and Explanation

The problem involves optimizing throughput by utilizing both GPU and CXL memory. Here's a step-by-step breakdown and conversion to LaTeX:

#### Hardware Parameters

1. **GPU Memory**:
   \[
   \text{GPU\_MEMORY} = 16 \times 1024^3 \text{ bytes} \quad (\text{16 GB})
   \]
2. **FLOPS (GPU)**:
   \[
   \text{FLOPS\_GPU} = 15.7 \times 10^{12} \text{ FLOPS} \quad (\text{15.7 TFLOPS})
   \]
3. **FLOPS (CXL)**:
   \[
   \text{FLOPS\_CXL} = 1.57 \times 10^{12} \text{ FLOPS} \quad (\text{1.57 TFLOPS})
   \]
4. **Bandwidth**:
   \[
   \text{BAND\_WIDTH} = 16 \times 1024^3 \text{ bytes/s} \quad (\text{16 GB/s})
   \]
5. **Latency**:
   \[
   \text{LATENCY} = 1 \times 10^{-6} \text{ seconds} \quad (\text{1 µs})
   \]

#### Model and Lora Dimensions

- **Model Dimension**:
  \[
  d = 512
  \]
- **Lora Dimension**:
  \[
  r = 4
  \]

#### Constraints and Time Computations

**Stage 1**:

1. **GPU Memory Usage**:
   \[
   4 \times (b1 + b2 + b3) \times d \leq \text{GPU\_MEMORY}
   \]
   where \( b1, b2, b3 \) are binary variables indicating which batches are processed.

2. **Stage 1 GPU Time**:
   \[
   T_{\text{GPU\_S1}} = \frac{2 \times b1 \times d \times r}{\text{FLOPS\_GPU}}
   \]

3. **Stage 1 CXL Time**:
   \[
   T_{\text{CXL\_S1}} = \text{LATENCY} + \frac{4 \times (b2 + b3) \times d}{\text{BAND\_WIDTH}} + \frac{2 \times (b2 + b3) \times d \times r}{\text{FLOPS\_CXL}} + \frac{4 \times b2 \times r}{\text{BAND\_WIDTH}}
   \]

4. **Constraints for Stage 1**:
   \[
   T_{\text{CXL\_S1}} \leq T_{\text{GPU\_S1}}
   \]
   \[
   4 \times b1 \times d + 4 \times d \times r + 4 \times b1 \times r \leq \text{GPU\_MEMORY}
   \]
   \[
   4 \times (b1 + b2) \times r \leq \text{GPU\_MEMORY}
   \]

**Stage 2**:

1. **Stage 2 GPU Time**:
   \[
   T_{\text{GPU\_S2}} = \frac{2 \times (b1 + b2) \times r \times d}{\text{FLOPS\_GPU}}
   \]

2. **Stage 2 CXL Time**:
   \[
   T_{\text{CXL\_S2}} = \frac{2 \times b3 \times r \times d}{\text{FLOPS\_CXL}} + \frac{4 \times b3 \times d}{\text{BAND\_WIDTH}} - \frac{4 \times b2 \times r}{\text{BAND\_WIDTH}}
   \]

3. **Constraints for Stage 2**:
   \[
   T_{\text{CXL\_S2}} \leq T_{\text{GPU\_S2}}
   \]
   \[
   4 \times (b1 + b2) \times r + 4 \times r \times d + 4 \times (b1 + b2) \times d \leq \text{GPU\_MEMORY}
   \]
   \[
   4 \times (b1 + b2 + b3) \times d \leq \text{GPU\_MEMORY}
   \]

#### Objective Function

Maximize throughput:
\[
\text{Throughput} = \frac{b1 + b2 + b3}{T_{\text{GPU\_S1}} + T_{\text{GPU\_S2}}}
\]
