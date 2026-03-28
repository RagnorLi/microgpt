"""
STEP 7：完整GPT（终点）
━━━━━━━━━━━━━━━━━━━━━━━
问题驱动（来自Step6）：
  ❌ 单头：同时只能从一个"角度"看信息（只有一种相关性）
     → 多头注意力：把n_embd切成n_head份，每份独立计算注意力，并行关注多种语义
  ❌ 无归一化：层数增加后数值爆炸
     → RMSNorm：把向量缩放到单位能量，防止信号随层数指数增长
  ❌ 无残差连接：梯度在多层间消失
     → 残差连接：x = x + f(x)，梯度可以绕过f(x)直接回流
  ❌ 注意力只"选择信息"，没有"独立思考"
     → MLP块：每个位置独立的"逐token思考"，expand→ReLU→compress

这就是完整的GPT-2架构（减去LayerNorm bias和绝对位置编码的小差异）。
和Karpathy的nano-gpt/makemore本质相同。

所有新概念：
  ✅ 多头注意力：把嵌入空间切片，每个头独立关注不同语义面
  ✅ RMSNorm：归一化层，稳定训练
  ✅ 残差连接：信息高速公路，解决深层网络梯度问题
  ✅ MLP块（FFN）：先扩张4×再压缩，提供非线性变换能力
  ✅ 多层堆叠：每层逐步精炼特征表示
  ✅ 温度采样：控制生成的"创造力"
"""

import os, math, random
random.seed(42)

if not os.path.exists('input.txt'):
    import urllib.request
    urllib.request.urlretrieve('https://raw.githubusercontent.com/karpathy/makemore/988aa59/names.txt', 'input.txt')

docs = [line.strip() for line in open('input.txt') if line.strip()]
random.shuffle(docs)
uchars = sorted(set(''.join(docs)))
BOS = len(uchars)
vocab_size = len(uchars) + 1
print(f"文档数: {len(docs)}, 词汇表: {vocab_size}")

# ── Autograd（不变，从Step3一路继承）────────────────────────
class Value:
    __slots__ = ('data', 'grad', '_children', '_local_grads')
    def __init__(self, data, children=(), local_grads=()):
        self.data = data; self.grad = 0
        self._children = children; self._local_grads = local_grads
    def __add__(self, o):
        o = o if isinstance(o, Value) else Value(o)
        return Value(self.data+o.data, (self,o), (1,1))
    def __mul__(self, o):
        o = o if isinstance(o, Value) else Value(o)
        return Value(self.data*o.data, (self,o), (o.data,self.data))
    def __pow__(self, n): return Value(self.data**n, (self,), (n*self.data**(n-1),))
    def log(self):        return Value(math.log(self.data), (self,), (1/self.data,))
    def exp(self):        return Value(math.exp(self.data), (self,), (math.exp(self.data),))
    def relu(self):       return Value(max(0,self.data), (self,), (float(self.data>0),))
    def __neg__(self):    return self * -1
    def __radd__(self, o): return self + o
    def __rmul__(self, o): return self * o
    def __truediv__(self, o): return self * o**-1
    def __sub__(self, o): return self + (-o)
    def backward(self):
        topo, visited = [], set()
        def build(v):
            if v not in visited:
                visited.add(v)
                for c in v._children: build(c)
                topo.append(v)
        build(self)
        self.grad = 1
        for v in reversed(topo):
            for child, lg in zip(v._children, v._local_grads):
                child.grad += lg * v.grad

# ── 超参数 ────────────────────────────────────────────────────
n_layer = 1      # Transformer层数（增加 = 更深，更强，更慢）
n_embd  = 16     # 嵌入维度（增加 = 更宽，表达力更强）
block_size = 16  # 最大上下文长度
n_head  = 4      # 注意力头数（n_embd必须能被n_head整除）
head_dim = n_embd // n_head  # 每个头的维度 = 16/4 = 4

# ── 工具函数 ──────────────────────────────────────────────────
def matrix(nout, nin, std=0.08):
    return [[Value(random.gauss(0, std)) for _ in range(nin)] for _ in range(nout)]

def linear(x, W):
    return [sum(wi * xi for wi, xi in zip(row, x)) for row in W]

def softmax(logits):
    m = max(v.data for v in logits)
    exps = [(v - m).exp() for v in logits]
    s = sum(exps)
    return [e / s for e in exps]

def rmsnorm(x):
    """RMS归一化：把向量缩放到单位能量，防止数值爆炸"""
    ms = sum(xi * xi for xi in x) / len(x)   # 均方
    scale = (ms + 1e-5) ** -0.5               # 1/√(均方)
    return [xi * scale for xi in x]

# ── 参数初始化 ────────────────────────────────────────────────
state_dict = {
    'wte':     matrix(vocab_size, n_embd),   # Token嵌入
    'wpe':     matrix(block_size, n_embd),   # 位置嵌入
    'lm_head': matrix(vocab_size, n_embd),   # 语言模型头
}
for i in range(n_layer):
    state_dict[f'layer{i}.attn_wq'] = matrix(n_embd, n_embd)   # Query投影
    state_dict[f'layer{i}.attn_wk'] = matrix(n_embd, n_embd)   # Key投影
    state_dict[f'layer{i}.attn_wv'] = matrix(n_embd, n_embd)   # Value投影
    state_dict[f'layer{i}.attn_wo'] = matrix(n_embd, n_embd)   # Output投影（合并多头）
    state_dict[f'layer{i}.mlp_fc1'] = matrix(4*n_embd, n_embd) # MLP扩张：16→64
    state_dict[f'layer{i}.mlp_fc2'] = matrix(n_embd, 4*n_embd) # MLP压缩：64→16

params = [p for mat in state_dict.values() for row in mat for p in row]
print(f"参数量: {len(params)}")

# ── GPT前向传播（完整版）────────────────────────────────────
def gpt(token_id, pos_id, keys, values):
    """
    token_id：当前输入token
    pos_id：当前位置
    keys/values：每层的KV Cache（列表的列表）
    返回：下一个token的logits
    """
    # 词嵌入 + 位置嵌入
    x = [t + p for t, p in zip(state_dict['wte'][token_id],
                                state_dict['wpe'][pos_id])]
    x = rmsnorm(x)  # 初始归一化（因为残差累积会改变量级）

    for li in range(n_layer):

        # ══ 多头注意力块 ══════════════════════════════════════
        x_res = x              # 保存残差（信息高速公路的"分叉口"）
        x = rmsnorm(x)         # 注意力前归一化

        # 计算Q、K、V（把x投影到三个不同的语义空间）
        q = linear(x, state_dict[f'layer{li}.attn_wq'])
        k = linear(x, state_dict[f'layer{li}.attn_wk'])
        v = linear(x, state_dict[f'layer{li}.attn_wv'])

        keys[li].append(k)     # 把当前K加入历史缓存
        values[li].append(v)   # 把当前V加入历史缓存

        x_attn = []
        for h in range(n_head):
            # 切片：每个头只看n_embd的一段（head_dim维）
            hs = h * head_dim
            q_h = q[hs:hs+head_dim]                          # 当前token的Query片段
            k_h = [ki[hs:hs+head_dim] for ki in keys[li]]   # 所有历史的Key片段
            v_h = [vi[hs:hs+head_dim] for vi in values[li]] # 所有历史的Value片段

            # 注意力分数：Q·K / √head_dim（缩放防饱和）
            attn_logits = [
                sum(q_h[j] * k_h[t][j] for j in range(head_dim)) / head_dim**0.5
                for t in range(len(k_h))
            ]
            attn_w = softmax(attn_logits)  # 注意力权重（概率分布）

            # 加权聚合Value
            head_out = [
                sum(attn_w[t] * v_h[t][j] for t in range(len(v_h)))
                for j in range(head_dim)
            ]
            x_attn.extend(head_out)   # 拼接所有头的输出

        x = linear(x_attn, state_dict[f'layer{li}.attn_wo'])  # 合并投影
        x = [a + b for a, b in zip(x, x_res)]  # 残差：x = 注意力输出 + 原始输入

        # ══ MLP块 ════════════════════════════════════════════
        x_res = x           # 再次保存残差
        x = rmsnorm(x)      # MLP前归一化
        x = linear(x, state_dict[f'layer{li}.mlp_fc1'])   # 扩张：n_embd → 4*n_embd
        x = [xi.relu() for xi in x]                        # 非线性激活
        x = linear(x, state_dict[f'layer{li}.mlp_fc2'])   # 压缩：4*n_embd → n_embd
        x = [a + b for a, b in zip(x, x_res)]  # 残差：x = MLP输出 + 注意力输出

    return linear(x, state_dict['lm_head'])  # 最终映射到词表

# ── Adam（和Step5/6相同）────────────────────────────────────
lr_base, beta1, beta2, eps_adam = 0.01, 0.85, 0.99, 1e-8
m_buf = [0.0] * len(params)
v_buf = [0.0] * len(params)

# ── 训练循环 ──────────────────────────────────────────────────
num_steps = 1000
for step in range(num_steps):
    doc = docs[step % len(docs)]
    tokens = [BOS] + [uchars.index(ch) for ch in doc] + [BOS]
    n = min(block_size, len(tokens) - 1)

    keys   = [[] for _ in range(n_layer)]   # 每个文档清空KV cache
    values = [[] for _ in range(n_layer)]
    losses = []

    for pos_id in range(n):
        token_id  = tokens[pos_id]
        target_id = tokens[pos_id + 1]
        logits = gpt(token_id, pos_id, keys, values)
        probs  = softmax(logits)
        losses.append(-probs[target_id].log())

    loss = (1 / n) * sum(losses)
    loss.backward()

    lr_t = lr_base * (1 - step / num_steps)
    t = step + 1
    for i, p in enumerate(params):
        g = p.grad
        m_buf[i] = beta1 * m_buf[i] + (1 - beta1) * g
        v_buf[i] = beta2 * v_buf[i] + (1 - beta2) * g * g
        mh = m_buf[i] / (1 - beta1 ** t)
        vh = v_buf[i] / (1 - beta2 ** t)
        p.data -= lr_t * mh / (vh ** 0.5 + eps_adam)
        p.grad = 0

    print(f"step {step+1:4d}/{num_steps} | loss {loss.data:.4f}", end='\r')

# ── 推理 ──────────────────────────────────────────────────────
temperature = 0.5
print("\n--- 生成的名字（完整GPT）---")
for idx in range(20):
    keys   = [[] for _ in range(n_layer)]
    values = [[] for _ in range(n_layer)]
    token_id = BOS
    name = []
    for pos_id in range(block_size):
        logits = gpt(token_id, pos_id, keys, values)
        probs  = softmax([l / temperature for l in logits])
        token_id = random.choices(range(vocab_size), weights=[p.data for p in probs])[0]
        if token_id == BOS:
            break
        name.append(uchars[token_id])
    print(f"sample {idx+1:2d}: {''.join(name)}")