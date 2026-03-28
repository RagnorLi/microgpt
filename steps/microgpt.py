"""
The most atomic way to train and run inference for a GPT in pure, dependency-free Python.
This file is the complete algorithm.
Everything else is just efficiency.

@karpathy
| 概念         | 第一性原理 (核心直觉)                                       |
| :----------- | :------------------------------------------------------ |
| 分词 (Token) | 语言 -> 整数，因为计算机只认数字                             |
| 嵌入 (Embed) | 整数 -> 坐标，意义变成空间中的位置                            |
| 注意力 (Attn) | Q问 K答 V给值，软性数据库检索                               |
| 残差 (Res)   | 梯度高速公路，允许"学增量"而非"学全量"                        |
| 自动微分 (AD) | 链式法则自动化，把"责任"从 Loss 倒推回每个参数                |
| 交叉熵 (Loss) | -log(正确词的概率)，概率越高越不惊讶，损失越低                |
| Adam 优化器  | 梯度方向稳 -> 走大步，梯度抖动 -> 走小步，自适应学习率          |
| 温度采样 (T)  | 不取最大值，按概率随机漫步 —— 这就是"创造力"的数学本质          |
"""
# ╔══════════════════════════════════════════════════════════════╗
# ║  深度注释版 · 费曼式直击灵魂注解                                  ║
# ║  核心真相：神经网络 = 用数字记住"什么跟着什么出现"的统计规律          ║
# ╚══════════════════════════════════════════════════════════════╝

import os       # os.path.exists  ← 检查文件是否已存在，避免重复下载
import math     # math.log, math.exp  ← 只用两个数学函数：对数（量化惊讶度）和指数（把分数变概率）
import random   # random.seed, random.choices, random.gauss, random.shuffle
random.seed(42) # 固定随机种子 ← 混沌中强加秩序，让实验可复现（42是宇宙答案，梗）

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 第一步：数据集  —— 模型的"人生经历"，没有经历就没有智慧
# 神经网络本质：用数据学"概率分布"，而非用规则编"if-else"
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
if not os.path.exists('input.txt'):         # 如果还没有数据文件
    import urllib.request                   # 临时引入网络下载工具
    names_url = 'https://raw.githubusercontent.com/karpathy/makemore/988aa59/names.txt'
    urllib.request.urlretrieve(names_url, 'input.txt')  # 下载3万个人名 ← 这就是模型的"全部人生"
docs = [line.strip() for line in open('input.txt') if line.strip()]  # 读成字符串列表，每个名字是一条"文档"
random.shuffle(docs)  # 打乱顺序 ← 防止模型死记硬背顺序，强迫它学规律而非序号
print(f"num docs: {len(docs)}")  # 打印文档总数，~32000个名字

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 第二步：分词器（Tokenizer）—— 把语言翻译成数字
# 真相：计算机不认识字母，只认识整数；分词 = 建立"字符↔整数"的双向词典
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
uchars = sorted(set(''.join(docs)))  # 所有文档拼在一起，找出不重复的字符 ← 这就是整个"字母表"（词汇表）
BOS = len(uchars)  # BOS = "Begin Of Sequence" ← 特殊符号，意思是"句子开始/结束"（用同一个token两用）
vocab_size = len(uchars) + 1  # 词汇表大小 = 所有字符 + 1个BOS ← 模型能认识的"词语"总数
print(f"vocab size: {vocab_size}")  # 打印词汇表大小，~27（26个字母+BOS）

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 第三步：自动微分（Autograd）—— 反向传播的灵魂
# 真相：学习 = 对错误求导；导数告诉你"每个参数往哪调才能减少错误"
# 链式法则 = 复合函数求导；自动微分 = 把链式法则自动化，不用手推
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class Value:
    __slots__ = ('data', 'grad', '_children', '_local_grads')  # 内存优化 ← 只允许这4个属性，省内存（Python默认用dict存属性，很慢）

    def __init__(self, data, children=(), local_grads=()):
        self.data = data                # 这个节点在前向传播中算出的"值"（标量）
        self.grad = 0                   # 损失对这个节点的导数 ← 初始为0，反向传播后被填充，代表"这个数该往哪调"
        self._children = children       # 生成这个节点的"父节点们" ← 记录计算图的拓扑结构
        self._local_grads = local_grads # 这个节点对其子节点的局部导数 ← 链式法则的"本地那一段"

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        # 加法节点：前向=a+b；反向：∂(a+b)/∂a=1, ∂(a+b)/∂b=1 ← 加法的梯度"原封不动往下传"
        return Value(self.data + other.data, (self, other), (1, 1))

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        # 乘法节点：前向=a*b；反向：∂(a*b)/∂a=b, ∂(a*b)/∂b=a ← 乘法梯度"互换因子"
        return Value(self.data * other.data, (self, other), (other.data, self.data))

    def __pow__(self, other): return Value(self.data**other, (self,), (other * self.data**(other-1),))
    # 幂函数：∂(x^n)/∂x = n*x^(n-1) ← 高中微积分，这里把它编码进计算图

    def log(self): return Value(math.log(self.data), (self,), (1/self.data,))
    # 对数：∂ln(x)/∂x = 1/x ← 用来计算交叉熵损失（量化"模型有多惊讶"）

    def exp(self): return Value(math.exp(self.data), (self,), (math.exp(self.data),))
    # 指数：∂e^x/∂x = e^x ← 神奇：导数等于自身！用在softmax里把分数变概率

    def relu(self): return Value(max(0, self.data), (self,), (float(self.data > 0),))
    # ReLU：负数归零，正数不变；梯度：正数=1（通过），负数=0（截断）
    # 本质：非线性激活函数 ← 没有它，多层网络等价于一层（纯线性叠加还是线性）

    def __neg__(self): return self * -1           # 取负 = 乘以-1
    def __radd__(self, other): return self + other # 支持 0 + Value（sum()从0开始累加需要这个）
    def __sub__(self, other): return self + (-other)   # 减法 = 加法 + 取负
    def __rsub__(self, other): return other + (-self)  # 反向减法
    def __rmul__(self, other): return self * other     # 支持 标量 * Value
    def __truediv__(self, other): return self * other**-1   # 除法 = 乘以倒数
    def __rtruediv__(self, other): return other * self**-1  # 反向除法

    def backward(self):
        # 反向传播的核心 ← "把责任从结果倒推回每一个参数"
        topo = []       # 存拓扑排序后的节点列表
        visited = set() # 防止重复访问同一节点
        def build_topo(v):
            # 深度优先搜索建立拓扑序 ← 确保"子节点在父节点之前处理"
            if v not in visited:
                visited.add(v)
                for child in v._children:
                    build_topo(child)    # 先递归处理所有子节点
                topo.append(v)           # 最后才加入自己（后序遍历）
        build_topo(self)                 # 从loss节点出发构建完整计算图
        self.grad = 1                    # 损失对损失自己的导数=1（起点）← dl/dl = 1
        for v in reversed(topo):        # 从loss倒着走回每个参数
            for child, local_grad in zip(v._children, v._local_grads):
                child.grad += local_grad * v.grad  # 链式法则：全局梯度 = 局部梯度 × 上游梯度
                # += 是因为一个节点可能被多条路径使用（梯度累加，不是覆盖）

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 第四步：初始化参数 —— 模型的"大脑"，知识就存在这些数字里
# 真相：GPT的所有"智慧"最终都是一张大表里的浮点数
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
n_layer = 1     # Transformer层数 ← 深度，每层学一种抽象级别的特征
n_embd = 16     # 嵌入维度 ← 宽度，每个token用16个数字来"表示自己的含义"
block_size = 16 # 最大上下文长度 ← 模型每次最多"看"多少个历史token（最长名字15字符，刚好）
n_head = 4      # 注意力头数 ← 把16维切成4份，每份独立关注不同"语义侧面"
head_dim = n_embd // n_head  # 每个头的维度 = 16/4 = 4 ← 多头注意力的分工粒度

matrix = lambda nout, nin, std=0.08: [[Value(random.gauss(0, std)) for _ in range(nin)] for _ in range(nout)]
# 初始化一个矩阵 ← 用高斯随机数（均值0，标准差0.08）初始化
# 为什么不全用0？全0梯度也全0，网络永远不会学；随机初始化"打破对称性"

state_dict = {
    'wte': matrix(vocab_size, n_embd),   # Token嵌入矩阵：把词id映射到向量 ← "词义空间"的坐标系
    'wpe': matrix(block_size, n_embd),   # Position嵌入矩阵：把位置id映射到向量 ← 告诉模型"我在第几个位置"
    'lm_head': matrix(vocab_size, n_embd) # 语言模型头：把最终向量映射回词表 ← "从含义空间解码回词语"
}
for i in range(n_layer):
    state_dict[f'layer{i}.attn_wq'] = matrix(n_embd, n_embd)  # Query矩阵 ← "我在问什么问题"
    state_dict[f'layer{i}.attn_wk'] = matrix(n_embd, n_embd)  # Key矩阵 ← "我能回答什么问题"
    state_dict[f'layer{i}.attn_wv'] = matrix(n_embd, n_embd)  # Value矩阵 ← "如果被选中，我贡献什么信息"
    state_dict[f'layer{i}.attn_wo'] = matrix(n_embd, n_embd)  # Output矩阵 ← 把多头结果合并投影回去
    state_dict[f'layer{i}.mlp_fc1'] = matrix(4 * n_embd, n_embd)  # MLP扩张层：16→64 ← 先"思考展开"
    state_dict[f'layer{i}.mlp_fc2'] = matrix(n_embd, 4 * n_embd)  # MLP压缩层：64→16 ← 再"提炼结论"

params = [p for mat in state_dict.values() for row in mat for p in row]
# 把所有矩阵摊平成一个大列表 ← 方便优化器统一遍历更新所有参数
print(f"num params: {len(params)}")  # 打印参数总量，~几千个（真实GPT-3是1750亿）

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 第五步：模型架构 —— 把token序列变成"下一个token的概率分布"
# 真相：语言模型 = 条件概率机器：P(下一个词 | 所有历史词)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def linear(x, w):
    # 线性变换 y = Wx ← 神经网络的基本操作，本质是"加权求和"
    # 每个输出=输入向量与权重矩阵某行的点积；点积=相似度测量
    return [sum(wi * xi for wi, xi in zip(wo, x)) for wo in w]

def softmax(logits):
    # 把任意实数分数变成"概率"（和为1，都为正）← 竞争机制：分越高，概率越大
    max_val = max(val.data for val in logits)  # 减最大值防数值溢出（e^很大数=爆炸）← 数值稳定技巧
    exps = [(val - max_val).exp() for val in logits]  # e^(xi - max) ← 都变成正数
    total = sum(exps)                                   # 求和作为分母
    return [e / total for e in exps]                    # 归一化 ← 让所有概率之和=1

def rmsnorm(x):
    # RMS归一化 ← 解决"梯度爆炸/消失"问题，让每层输入都在合理数值范围
    # 本质：把向量缩放到"单位能量"，防止信号随层数指数级放大或缩小
    ms = sum(xi * xi for xi in x) / len(x)  # 计算均方（Mean Square）← 向量能量
    scale = (ms + 1e-5) ** -0.5              # 1/√(均方+ε) ← 归一化因子，ε防止除零
    return [xi * scale for xi in x]          # 每个元素除以RMS ← 向量能量归一

def gpt(token_id, pos_id, keys, values):
    # 整个GPT的一次前向传播 ← 输入：当前token + 位置；输出：下一个token的概率分布（logits）
    # 注意：这是"自回归"模型，每次只处理一个token，但能"看到"所有历史（通过KV cache）

    tok_emb = state_dict['wte'][token_id]  # 查表：token_id → 16维向量 ← "把词ID翻译成坐标"
    pos_emb = state_dict['wpe'][pos_id]    # 查表：位置id → 16维向量 ← "把位置翻译成坐标"
    x = [t + p for t, p in zip(tok_emb, pos_emb)]  # 词义 + 位置 = 完整的"此时此地的含义"
    x = rmsnorm(x)  # 归一化 ← 此处非多余：残差连接会导致x的量级积累，需要在此重置

    for li in range(n_layer):
        # ── 注意力块：让每个位置"看到"并"整合"其他位置的信息 ──
        x_residual = x   # 保存残差 ← 残差连接="信息高速公路"，允许梯度绕过复杂变换直接回流
        x = rmsnorm(x)   # 注意力前归一化 ← 让输入处于稳定范围

        q = linear(x, state_dict[f'layer{li}.attn_wq'])  # Query = "我这个token想查询什么信息"
        k = linear(x, state_dict[f'layer{li}.attn_wk'])  # Key   = "我这个token能提供什么信息"
        v = linear(x, state_dict[f'layer{li}.attn_wv'])  # Value = "如果我被选中，我传递什么内容"
        # Q/K/V 的本质：注意力 = 软性的信息检索，Q是查询，K是索引，V是数据库内容

        keys[li].append(k)    # 把当前token的Key加入历史缓存 ← KV Cache：避免重复计算历史token
        values[li].append(v)  # 把当前token的Value加入历史缓存

        x_attn = []  # 存放所有注意力头的输出
        for h in range(n_head):
            hs = h * head_dim  # 当前头的起始下标
            q_h = q[hs:hs+head_dim]                        # 切出当前头的Query片段（4维）
            k_h = [ki[hs:hs+head_dim] for ki in keys[li]] # 所有历史token的Key片段
            v_h = [vi[hs:hs+head_dim] for vi in values[li]] # 所有历史token的Value片段

            attn_logits = [
                sum(q_h[j] * k_h[t][j] for j in range(head_dim)) / head_dim**0.5
                for t in range(len(k_h))
            ]
            # 注意力分数 = Q·K / √d ← 点积衡量"Query和Key有多匹配"
            # 除以√d：防止维度高时点积过大导致softmax饱和（梯度消失）

            attn_weights = softmax(attn_logits)
            # 把分数变成概率 ← "我应该把多少注意力分配给历史上每个位置"

            head_out = [
                sum(attn_weights[t] * v_h[t][j] for t in range(len(v_h)))
                for j in range(head_dim)
            ]
            # 加权求和Values ← 注意力的最终输出：按重要性混合所有历史信息
            # 本质：软性的"从记忆中提取相关信息"

            x_attn.extend(head_out)  # 拼接每个头的输出 ← 多头 = 从多个"语义侧面"同时关注

        x = linear(x_attn, state_dict[f'layer{li}.attn_wo'])  # 投影回原始维度 ← 融合多头信息
        x = [a + b for a, b in zip(x, x_residual)]  # 残差连接 ← x = 注意力输出 + 原始输入
        # 残差的意义：即使注意力学到的是"什么都不改变"，梯度仍能无障碍回流

        # ── MLP块：在每个位置独立做"深度思考"──
        x_residual = x         # 再次保存残差
        x = rmsnorm(x)         # MLP前归一化
        x = linear(x, state_dict[f'layer{li}.mlp_fc1'])  # 16→64：先"展开思维"，进入高维特征空间
        x = [xi.relu() for xi in x]   # ReLU非线性 ← 没有这步，MLP只是线性变换，叠多少层都没用
        x = linear(x, state_dict[f'layer{li}.mlp_fc2'])  # 64→16：再"压缩结论"，提炼有用信息
        x = [a + b for a, b in zip(x, x_residual)]  # 残差连接 ← 同上，让网络学"增量修正"而非"全量重写"

    logits = linear(x, state_dict['lm_head'])
    # 最终把向量投影回词表空间 ← 每个词一个分数，"哪个词更可能是下一个"
    return logits

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 第六步：Adam优化器 —— 比普通梯度下降更聪明地更新参数
# 普通SGD：所有参数学习率相同；Adam：每个参数自适应学习率
# 直觉：步子的大小根据历史梯度动态调整，震荡维度走小步，平稳维度走大步
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
learning_rate, beta1, beta2, eps_adam = 0.01, 0.85, 0.99, 1e-8
# learning_rate=0.01：全局步长；beta1=0.85：一阶动量衰减（梯度的"惯性"）；
# beta2=0.99：二阶动量衰减（梯度平方的"历史方差"）；eps=1e-8：防除零

m = [0.0] * len(params)  # 一阶矩（梯度的指数移动平均）← 动量，记住"过去往哪走的"
v = [0.0] * len(params)  # 二阶矩（梯度平方的指数移动平均）← 方差，记住"过去走得有多不稳定"

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 第七步：训练循环 —— 机器学习的本质：重复"犯错→求责任→修正"
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
num_steps = 1000  # 训练步数 ← 总共"学习"1000次（每次喂一个名字）
for step in range(num_steps):

    # ── 取一条数据，转化为token序列 ──
    doc = docs[step % len(docs)]  # 循环取文档 ← step % len(docs)确保不越界
    tokens = [BOS] + [uchars.index(ch) for ch in doc] + [BOS]
    # BOS开头+字符token序列+BOS结尾 ← BOS同时充当"开始信号"和"停止信号"，优雅
    n = min(block_size, len(tokens) - 1)  # 序列长度（最多block_size）← 防止超出上下文窗口

    # ── 前向传播：计算损失 ──
    keys, values = [[] for _ in range(n_layer)], [[] for _ in range(n_layer)]
    # 清空KV Cache ← 每个新文档是独立的，不能把上条名字的记忆带过来
    losses = []
    for pos_id in range(n):
        token_id, target_id = tokens[pos_id], tokens[pos_id + 1]
        # 输入当前token，目标是预测下一个token ← 语言模型训练的核心范式："预测下一个词"
        logits = gpt(token_id, pos_id, keys, values)  # 前向传播 → 得到logits
        probs = softmax(logits)                         # logits → 概率分布
        loss_t = -probs[target_id].log()               # 负对数似然 ← 交叉熵损失
        # 本质：-log(正确词的概率)；概率越高损失越低；模型越确信正确答案，损失越趋近0
        losses.append(loss_t)
    loss = (1 / n) * sum(losses)  # 对整个序列求平均损失 ← 一个文档贡献一个标量损失值

    # ── 反向传播：计算每个参数的梯度 ──
    loss.backward()
    # 一行代码，完成整个计算图的链式法则反向传播
    # 结果：state_dict里每个参数p的p.grad被填上"它对loss的贡献有多大"

    # ── Adam更新参数 ──
    lr_t = learning_rate * (1 - step / num_steps)  # 线性学习率衰减 ← 训练末期走小步，精细调整
    for i, p in enumerate(params):
        m[i] = beta1 * m[i] + (1 - beta1) * p.grad
        # 一阶矩更新：新动量 = 0.85×旧动量 + 0.15×当前梯度 ← 梯度的加权移动平均（平滑抖动）
        v[i] = beta2 * v[i] + (1 - beta2) * p.grad ** 2
        # 二阶矩更新：新方差 = 0.99×旧方差 + 0.01×梯度平方 ← 梯度大小的历史记录
        m_hat = m[i] / (1 - beta1 ** (step + 1))  # 偏差修正 ← 初始阶段m/v被低估，乘以修正因子
        v_hat = v[i] / (1 - beta2 ** (step + 1))  # 偏差修正（同上）
        p.data -= lr_t * m_hat / (v_hat ** 0.5 + eps_adam)
        # 参数更新 ← 核心公式：步长 = 学习率 × 动量均值 / √方差
        # 直觉：梯度方向一致（大动量）且稳定（小方差）→ 走大步；方向乱（小动量）或抖（大方差）→ 走小步
        p.grad = 0  # 清零梯度 ← 必须！否则下次反向传播会在旧梯度上累加

    print(f"step {step+1:4d} / {num_steps:4d} | loss {loss.data:.4f}", end='\r')
    # 打印训练进度 ← loss应该从~3.3（随机猜测≈ln(27)）逐渐下降

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 第八步：推理（Inference）—— 让模型"发挥"，从学到的分布中采样
# 真相：推理不是检索记忆，而是从学到的"概率地图"中随机漫步
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
temperature = 0.5  # 温度参数（0,1] ← 控制"创造力"：越低越保守（选高概率），越高越随机（探索低概率）
# 本质：logits /= temperature；temperature→0时变成贪婪解码；temperature→∞时变成均匀随机
print("\n--- inference (new, hallucinated names) ---")
for sample_idx in range(20):  # 生成20个名字
    keys, values = [[] for _ in range(n_layer)], [[] for _ in range(n_layer)]  # 清空KV cache
    token_id = BOS   # 以BOS开始 ← 告诉模型"一个新序列要开始了"
    sample = []      # 存生成的字符
    for pos_id in range(block_size):  # 最多生成block_size个字符
        logits = gpt(token_id, pos_id, keys, values)  # 前向传播得到下一个token的logits
        probs = softmax([l / temperature for l in logits])  # 用温度缩放后取softmax
        token_id = random.choices(range(vocab_size), weights=[p.data for p in probs])[0]
        # 按概率分布随机采样 ← 不是取最大值！随机采样才能生成多样化输出
        if token_id == BOS:
            break  # 采样到BOS = 模型认为"名字结束了" ← BOS兼任EOS（结束符），优雅的设计
        sample.append(uchars[token_id])  # 把token_id翻译回字符
    print(f"sample {sample_idx+1:2d}: {''.join(sample)}")  # 打印生成的名字
    # 输出是模型幻觉出的"新名字"：从未出现过，但符合训练数据的统计规律
    # 这就是生成式AI的本质：不是复制，而是在学到的概率空间里"随机漫步"出新内容