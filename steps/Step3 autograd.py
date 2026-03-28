"""
STEP 3：加入自动微分（Autograd）
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
问题驱动：Step2手推梯度勉强能用，但加一个隐藏层就爆炸。
我需要一个系统能自动帮我求任意计算图的梯度。

核心洞察：把每个数包装成Value对象，记录"我是怎么算出来的"
  - 前向传播：正常计算，同时建立计算图
  - 反向传播：从loss出发，沿计算图逆向用链式法则
  - 这就是所有深度学习框架（PyTorch/JAX）的核心原理

新增概念：
  ✅ 计算图：有向无环图，节点=数值，边=运算关系
  ✅ 链式法则：∂loss/∂x = ∂loss/∂y × ∂y/∂x （本地梯度 × 上游梯度）
  ✅ 拓扑排序：确保反向传播时"上游节点先于下游节点"被处理
  ✅ loss.backward()：一行代码，完成整个计算图的梯度计算

架构：仍然是二元模型（和Step2完全相同），只是梯度改为自动计算
这让你看清：autograd是独立工具，和模型架构无关

遗留问题（驱动你进入Step 4）：
  ❌ 仍然只有1个字符的上下文
  ❌ 需要更多上下文才能生成合理的名字
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
# 自动微分引擎（从零实现）
# 每个Value节点存：数值、梯度、子节点、局部梯度
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class Value:
    def __init__(self, data, children=(), local_grads=()):
        self.data = data
        self.grad = 0
        self._children = children
        self._local_grads = local_grads

    # 定义运算，同时记录"这个数是怎么来的"
    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return Value(self.data + other.data, (self, other), (1, 1))
        # 加法：∂(a+b)/∂a=1, ∂(a+b)/∂b=1

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return Value(self.data * other.data, (self, other), (other.data, self.data))
        # 乘法：∂(a*b)/∂a=b, ∂(a*b)/∂b=a

    def __pow__(self, n): return Value(self.data**n, (self,), (n * self.data**(n-1),))
    def log(self):        return Value(math.log(self.data), (self,), (1/self.data,))
    def exp(self):        return Value(math.exp(self.data), (self,), (math.exp(self.data),))
    def __neg__(self):    return self * -1
    def __radd__(self, o): return self + o
    def __rmul__(self, o): return self * o
    def __truediv__(self, o): return self * o**-1
    def __sub__(self, o): return self + (-o)

    def backward(self):
        # 建立拓扑序（深度优先后序遍历）
        topo, visited = [], set()
        def build(v):
            if v not in visited:
                visited.add(v)
                for c in v._children: build(c)
                topo.append(v)
        build(self)

        self.grad = 1   # ∂loss/∂loss = 1
        for v in reversed(topo):
            for child, lg in zip(v._children, v._local_grads):
                child.grad += lg * v.grad  # 链式法则

# ── 模型参数（Value对象组成的矩阵）────────────────────────────
W = [[Value(random.gauss(0, 0.01)) for _ in range(vocab_size)] for _ in range(vocab_size)]
params = [p for row in W for p in row]

def softmax(logits):
    """把Value列表变成概率（数值稳定）"""
    max_val = max(v.data for v in logits)
    exps = [(v - max_val).exp() for v in logits]
    total = sum(exps)
    return [e / total for e in exps]

# ── 训练循环 ──────────────────────────────────────────────────
lr = 0.1
for step in range(1000):
    doc = docs[step % len(docs)]
    tokens = [BOS] + [uchars.index(ch) for ch in doc] + [BOS]
    n = len(tokens) - 1

    losses = []
    for a, b in zip(tokens, tokens[1:]):
        probs = softmax(W[a])          # 前向：算概率
        loss_t = -probs[b].log()       # 交叉熵（现在是Value，会自动建计算图）
        losses.append(loss_t)

    loss = sum(losses) * (1/n)         # 平均损失

    # 自动反向传播！一行代码搞定所有梯度
    loss.backward()

    # SGD更新 + 清零梯度
    for p in params:
        p.data -= lr * p.grad
        p.grad = 0

    if (step + 1) % 200 == 0:
        print(f"step {step+1:4d} | loss {loss.data:.4f}")

# ── 推理 ──────────────────────────────────────────────────────
print("\n--- 生成的名字（autograd 二元网络）---")
for _ in range(10):
    token = BOS
    name = []
    for _ in range(20):
        probs = softmax(W[token])
        token = random.choices(range(vocab_size), weights=[p.data for p in probs])[0]
        if token == BOS: break
        name.append(uchars[token])
    print(''.join(name))