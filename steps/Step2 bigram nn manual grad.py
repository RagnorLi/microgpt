"""
STEP 2：二元神经网络 + 手动梯度 + SGD
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
问题驱动：Step1的概率表无法"优化"，只能计数。
我想要一个可以通过梯度下降学习的模型。
 
核心洞察：把概率表替换成一个可训练的权重矩阵W
  - 输入token（整数）→ W的某一行 → softmax → 概率分布
  - 计算损失（交叉熵）→ 手动推导梯度 → SGD更新W
  - 这就是最简单的神经网络：一层线性 + softmax
 
新增概念：
  ✅ 损失函数：交叉熵 = -log(正确类别的概率)
  ✅ 梯度：dL/dW[i][j] = probs[j] - (j==target)  ← softmax+交叉熵的梯度公式
  ✅ SGD更新：W -= lr * grad
 
遗留问题（驱动你进入Step 3）：
  ❌ 手动推导梯度只对简单模型可行
  ❌ 加一个隐藏层，梯度就要手推两页纸
  ❌ 需要自动微分
"""

import random, math
random.seed(42)

# ── 数据 ──────────────────────────────────────────────────────
docs = [line.strip() for line in open('input.txt') if line.strip()]
random.shuffle(docs)
uchars = sorted(set(''.join(docs)))
BOS = len(uchars)
vocab_size = len(uchars) + 1


# ── 模型参数：一个可训练的权重矩阵 ───────────────────────────
# W[i][j] = "当前token是i时，下一个token是j的分数（logit）"
# 初始化为小随机数（不用全0：全0梯度对称，永远学不动）
W = [[random.gauss(0, 0.01) for _ in range(vocab_size)] for _ in range(vocab_size)]
 
def softmax(logits):
    """把logits变成概率（数值稳定版）"""
    m = max(logits)
    exps = [math.exp(x - m) for x in logits]
    s = sum(exps)
    return [e / s for e in exps]

# ── 训练循环 ──────────────────────────────────────────────────
learning_rate = 0.1
for step in range(1000):
    doc = docs[step % len(docs)]
    tokens = [BOS] + [uchars.index(ch) for ch in doc] + [BOS]
 
    total_loss = 0.0
    # 梯度缓冲区（每步清零）
    dW = [[0.0] * vocab_size for _ in range(vocab_size)]
    n = len(tokens) - 1
 
    for a, b in zip(tokens, tokens[1:]):    # a=当前, b=目标（下一个）
        probs = softmax(W[a])               # 前向：W的第a行 → 概率分布
        loss = -math.log(probs[b] + 1e-10)  # 交叉熵损失
        total_loss += loss
 
        # 手动计算梯度（softmax+交叉熵的解析解）
        # dL/d(W[a][j]) = probs[j] - 1(j==b)
        # 直觉：模型高估了非目标类 → 把它们的权重往下调；低估了目标类 → 往上调
        for j in range(vocab_size):
            dW[a][j] += probs[j] - (1.0 if j == b else 0.0)
 
    # SGD：沿梯度反方向更新
    for i in range(vocab_size):
        for j in range(vocab_size):
            W[i][j] -= learning_rate * dW[i][j] / n
 
    if (step + 1) % 200 == 0:
        print(f"step {step+1:4d} | loss {total_loss/n:.4f}")
         

# ── 推理（自回归采样）─────────────────────────────────────────
print("\n--- 生成的名字（二元神经网络）---")
for _ in range(10):
    token = BOS
    name = []
    for _ in range(20):
        probs = softmax(W[token])
        token = random.choices(range(vocab_size), weights=probs)[0]
        if token == BOS:
            break
        name.append(uchars[token])
    print(''.join(name))