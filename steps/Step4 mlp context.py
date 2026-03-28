"""
STEP 4：MLP + 多字符上下文窗口（Bengio 2003）
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
问题驱动：Step3只看1个字符。"emm" 和 "ama" 后面应该接什么不同，
但模型只看最后1个字符，完全区分不了。
需要让模型看过去N个字符的上下文。

核心洞察：把最近 CONTEXT_SIZE 个字符的嵌入向量拼在一起，
喂进一个两层MLP，预测下一个字符。
这就是2003年Bengio的神经网络语言模型论文的思想。

新增概念：
  ✅ 嵌入（Embedding）：把整数token映射到稠密向量，学习"字符含义"
  ✅ 向量拼接（Concatenate）：把多个嵌入拼成一个长向量作为MLP输入
  ✅ 隐藏层 + ReLU：非线性激活，让网络能学非线性规律
  ✅ 两层MLP：input → hidden → output
  ✅ 参数量：现在有成百上千个参数，autograd价值体现

遗留问题（驱动你进入Step 5）：
  ❌ SGD在参数多、Loss面复杂时收敛慢、抖动大
  ❌ 需要更智能的优化器
"""

import random, math
random.seed(42)

# ── 数据 ──────────────────────────────────────────────────────
docs = [line.strip() for line in open('input.txt') if line.strip()]
random.shuffle(docs)
uchars = sorted(set(''.join(docs)))
BOS = len(uchars)
vocab_size = len(uchars) + 1

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Autograd（从Step3复制，完全不变）
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
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

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 模型参数：嵌入表 + MLP两层
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CONTEXT_SIZE = 2   # 看过去几个字符
n_embd = 2        # 每个字符的嵌入维度
n_hidden = 10      # 隐藏层宽度

def matrix(nout, nin, std=0.1):
    return [[Value(random.gauss(0, std)) for _ in range(nin)] for _ in range(nout)]

# 嵌入表：vocab_size 个字符，每个用 n_embd 维向量表示
emb_table = matrix(vocab_size, n_embd)

# MLP：输入 = CONTEXT_SIZE个嵌入拼在一起 = CONTEXT_SIZE * n_embd
# 隐藏层：输入维 → n_hidden，输出层：n_hidden → vocab_size
W1 = matrix(n_hidden, CONTEXT_SIZE * n_embd)
W2 = matrix(vocab_size, n_hidden)

params = ([p for row in emb_table for p in row] +
          [p for row in W1 for p in row] +
          [p for row in W2 for p in row])
print(f"参数量: {len(params)}")  # 比Step3多很多，手推梯度不现实了

# ── 工具函数 ──────────────────────────────────────────────────
def softmax(logits):
    m = max(v.data for v in logits)
    exps = [(v - m).exp() for v in logits]
    s = sum(exps)
    return [e / s for e in exps]

def linear(x, W):
    """矩阵乘法：y = Wx"""
    return [sum(wi * xi for wi, xi in zip(w_row, x)) for w_row in W]

# ── 训练循环 ──────────────────────────────────────────────────
lr = 0.05
for step in range(5000):
    doc = docs[step % len(docs)]
    tokens = [BOS] + [uchars.index(ch) for ch in doc] + [BOS]

    # 构建输入序列：每个位置用前CONTEXT_SIZE个token作为上下文
    # 用BOS填充（序列开头没有足够的历史）
    context = [BOS] * CONTEXT_SIZE
    losses = []

    for target in tokens[1:]:   # 目标：序列中每个token（从第2个开始）
        # 把上下文中每个token的嵌入向量拼成一个长向量
        x = []
        for tok in context:
            x.extend(emb_table[tok])   # 拼接嵌入

        # MLP前向传播
        h = linear(x, W1)              # 第一层线性
        h = [xi.relu() for xi in h]    # 非线性激活
        logits = linear(h, W2)         # 第二层线性 → logits

        # 损失
        probs = softmax(logits)
        losses.append(-probs[target].log())

        # 滑动上下文窗口
        context = context[1:] + [target]

    n = len(losses)
    loss = sum(losses) * (1/n)
    loss.backward()

    for p in params:
        p.data -= lr * p.grad
        p.grad = 0

    if (step + 1) % 1000 == 0:
        print(f"step {step+1:5d} | loss {loss.data:.4f}")

# ── 推理 ──────────────────────────────────────────────────────
print("\n--- 生成的名字（MLP + 上下文窗口）---")
for _ in range(10):
    context = [BOS] * CONTEXT_SIZE
    name = []
    for _ in range(20):
        x = []
        for tok in context:
            x.extend(emb_table[tok])
        h = [xi.relu() for xi in linear(x, W1)]
        logits = linear(h, W2)
        probs = softmax(logits)
        token = random.choices(range(vocab_size), weights=[p.data for p in probs])[0]
        if token == BOS: break
        name.append(uchars[token])
        context = context[1:] + [token]
    print(''.join(name))