"""
STEP 6：单头自注意力（Self-Attention）
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
问题驱动：Step5的MLP对所有位置同等对待。
生成"emma"时，看"em"和看"mm"后面接"a"的概率应该不同，
但MLP拿到的是固定窗口内所有位置的平均信息，无法"有选择地"关注。

核心洞察：注意力 = 软性的信息检索
  数据库比喻：
    - Query（Q）= "我想查什么？"
    - Key（K）  = "我这条记录是关于什么的？"
    - Value（V）= "如果你选中我，我给你什么信息？"
    - 注意力权重 = softmax(Q·K / √d) → "每条历史记录有多相关"
    - 输出 = Σ(权重 × V) → 按相关性加权平均所有历史信息

新增概念：
  ✅ Token嵌入（wte）：词ID → 向量
  ✅ 位置嵌入（wpe）：位置ID → 向量（告诉模型"我在第几个位置"）
  ✅ Q/K/V投影矩阵：把同一个x投影到三个不同"语义空间"
  ✅ 缩放点积注意力：Q·Kᵀ / √head_dim（防止高维时点积过大）
  ✅ KV Cache：推理时缓存历史K/V，避免重复计算
  ✅ 自回归模式：每次只传入一个token，用KV cache访问历史

遗留问题（驱动你进入Step 7）：
  ❌ 单头注意力同时只能关注一种"语义关系"
  ❌ 没有残差连接：梯度在深层网络中容易消失
  ❌ 没有归一化：数值不稳定
  ❌ 注意力之后缺少"独立思考"的MLP块
"""

import random, math
random.seed(42)

# ── 数据 ──────────────────────────────────────────────────────
docs = [line.strip() for line in open('input.txt') if line.strip()]
random.shuffle(docs)
uchars = sorted(set(''.join(docs)))
BOS = len(uchars)
vocab_size = len(uchars) + 1

# ── Autograd（不变）──────────────────────────────────────────
class Value:
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

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 模型参数（新增：位置嵌入，Q/K/V/O矩阵）
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
n_embd = 16
block_size = 16

wte = matrix(vocab_size, n_embd)   # Token嵌入：vocab_size × n_embd
wpe = matrix(block_size, n_embd)   # 位置嵌入：block_size × n_embd
# 注意力投影矩阵：把x（n_embd维）分别投影到Q、K、V空间
Wq = matrix(n_embd, n_embd)        # Query矩阵
Wk = matrix(n_embd, n_embd)        # Key矩阵
Wv = matrix(n_embd, n_embd)        # Value矩阵
Wo = matrix(n_embd, n_embd)        # Output投影：把注意力输出映射回n_embd
lm_head = matrix(vocab_size, n_embd)  # 语言模型头：n_embd → vocab_size（logits）

params = ([p for row in wte for p in row] +
          [p for row in wpe for p in row] +
          [p for row in Wq for p in row] +
          [p for row in Wk for p in row] +
          [p for row in Wv for p in row] +
          [p for row in Wo for p in row] +
          [p for row in lm_head for p in row])
print(f"参数量: {len(params)}")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 单头自注意力前向传播（自回归模式，带KV Cache）
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def forward(token_id, pos_id, kv_cache_k, kv_cache_v):
    """
    输入：当前token_id + 位置pos_id + 历史KV缓存
    输出：下一个token的logits
    kv_cache_k/v：列表，存储所有历史token的K/V向量
    """
    # 1. 嵌入：token含义 + 位置信息
    tok_emb = wte[token_id]
    pos_emb = wpe[pos_id]
    x = [t + p for t, p in zip(tok_emb, pos_emb)]  # 加在一起

    # 2. 计算当前token的Q、K、V
    q = linear(x, Wq)  # Query："我想问什么问题"
    k = linear(x, Wk)  # Key："我能回答什么"
    v = linear(x, Wv)  # Value："如果被选中，我贡献什么"

    # 3. 更新KV Cache（把当前token的K/V加入历史）
    kv_cache_k.append(k)
    kv_cache_v.append(v)

    # 4. 计算注意力：当前Q与所有历史K的点积相似度
    scale = n_embd ** 0.5  # 缩放因子：防止高维时点积过大
    attn_logits = [
        sum(q[j] * kv_cache_k[t][j] for j in range(n_embd)) / scale
        for t in range(len(kv_cache_k))
    ]
    # attn_logits[t] = "当前token对历史位置t的关注程度（原始分数）"

    # 5. softmax → 注意力权重（概率分布，和为1）
    attn_weights = softmax(attn_logits)

    # 6. 按权重加权平均所有历史的Value
    attn_out = [
        sum(attn_weights[t] * kv_cache_v[t][j] for t in range(len(kv_cache_v)))
        for j in range(n_embd)
    ]
    # attn_out：融合了"所有历史信息，按相关性加权"的向量

    # 7. 输出投影 + 语言模型头
    x_out = linear(attn_out, Wo)       # 把多维注意力输出投影回n_embd
    logits = linear(x_out, lm_head)    # 映射到词表：每个词一个分数
    return logits

# ── Adam优化器（和Step5相同）─────────────────────────────────
lr_base, beta1, beta2, eps = 0.01, 0.9, 0.99, 1e-8
adam_m = [0.0] * len(params)
adam_v_buf = [0.0] * len(params)

# ── 训练循环 ──────────────────────────────────────────────────
num_steps = 1000
for step in range(num_steps):
    doc = docs[step % len(docs)]
    tokens = [BOS] + [uchars.index(ch) for ch in doc] + [BOS]
    n = min(block_size, len(tokens) - 1)

    kv_k, kv_v = [], []   # KV cache：每个文档清空
    losses = []

    for pos_id in range(n):
        token_id = tokens[pos_id]
        target_id = tokens[pos_id + 1]

        logits = forward(token_id, pos_id, kv_k, kv_v)
        probs = softmax(logits)
        losses.append(-probs[target_id].log())

    loss = sum(losses) * (1 / len(losses))
    loss.backward()

    lr_t = lr_base * (1 - step / num_steps)
    t = step + 1
    for i, p in enumerate(params):
        g = p.grad
        adam_m[i] = beta1 * adam_m[i] + (1 - beta1) * g
        adam_v_buf[i] = beta2 * adam_v_buf[i] + (1 - beta2) * g * g
        m_hat = adam_m[i] / (1 - beta1 ** t)
        v_hat = adam_v_buf[i] / (1 - beta2 ** t)
        p.data -= lr_t * m_hat / (v_hat ** 0.5 + eps)
        p.grad = 0

    if (step + 1) % 200 == 0:
        print(f"step {step+1:4d} | loss {loss.data:.4f}")

# ── 推理 ──────────────────────────────────────────────────────
print("\n--- 生成的名字（单头注意力）---")
temperature = 0.5
for _ in range(10):
    kv_k, kv_v = [], []
    token_id = BOS
    name = []
    for pos_id in range(block_size):
        logits = forward(token_id, pos_id, kv_k, kv_v)
        probs = softmax([l / temperature for l in logits])
        token_id = random.choices(range(vocab_size), weights=[p.data for p in probs])[0]
        if token_id == BOS: break
        name.append(uchars[token_id])
    print(''.join(name))