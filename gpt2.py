"""
know the whole damn thing about llm by chinese idioms pattern find and generate by gpt2(nn) 
it's so damn good 

insight: 
1、整个文件就是一个函数复合 L(θ) = -log p_θ(next | context)，然后用链式法则求导 ∇_θ L，再用Adam去优化爬坡。其余一切都是这个的展开。
2、整个 GPT 的 tensor flow 只有 一个不变量：中间表示始终是 [B, T, C]。Attention 内部短暂变形为 [B, nh, T, hs] 做路由， MLP 内部短暂膨胀到 [B, T, 4C] 做处理，但 输入输出永远回到 [B, T, C]。 这个维度守恒约束让残差连接成为可能，让层可以任意堆叠。
"""
import enum
import numpy as np
import os, math
import random

from sympy.geometry import line
from torch.cuda import temperature
random.seed(42)

# ① Load the raw dataset
# docs: list of strings, shuffled for stochasticity
if not os.path.exists('input.txt'):
    import urllib.request
    names_url = 'https://raw.githubusercontent.com/karpathy/makemore/988aa59/names.txt'
    urllib.request.urlretrieve(names_url, 'input.txt')
docs = [line.strip() for line in open('input.txt') if line.strip()]
random.shuffle(docs)
print(f"num docs: {len(docs)}")

# ② Tokenizer: bijection between characters and integers
# ucharts -> vocab; BOS as sentinel just-seperate token; vocab_size = len(uchars) + 1
ucharts = sorted(set(''.join(docs)))  # 3808 unique zi
BOS = len(ucharts)
vocab_size = len(ucharts) + 1
print(f"vocab size: {vocab_size}")


# ③ Autograd: build a DAG of Value nodes during forward pass,
# then reverse-topological-sort to propagate chain rule backward
# Each node stores: data, grad, children, local_grads
class Value:
    """
    f′(x)=h→0lim​hf(x+h)−f(x)​
    Python 的运算符是双向派发。左边先试，失败了右边接。
    """
    __slots__ = ('data', 'grad', '_children', '_local_grads')  # Python optimization for memory usage

    def __init__(self, data, children=(), local_grad=()):
        self.data = data  # scalar value of this node calculated during forward pass
        self.grad = 0  # derivative of the loss w.r.t. this node, caculated in backward pass
        self._children = children  # children of this node in the computation graph
        self._local_grads = local_grad # local derivative of this node w.r.t. it's children

    def __add__(self, other):
        """ + """
        other = other if isinstance(other, Value) else Value(other)  # 💣 karpathy designed it this way intentionally to only handle scalars
        return Value(self.data + other.data, (self, other), (1, 1))

    def __mul__(self, other):
        """ * """
        other = other if isinstance(other, Value) else Value(other)
        return Value(self.data * other.data, (self, other), (other.data, self.data))    
    
    def __pow__(self, other):
        """ self.data ^ other"""
        return Value(self.data ** other, (self, ), (other * self.data ** (other - 1), ))

    def __neg__(self):
        """ -self.data """
        return self * -1  # triggle ↑ * so return Value type 
    
    def __sub__(self, other):
        """ - """
        return self + (-other)

    def __truediv__(self, other):
        """ self / other """
        return self * other**-1  
    
    def __rtruediv__(self, other):
        """ other / self 
        
        解释下python的运算符双向派发，以及karpathy极简但深刻的一点点
        # 原始调用: (3).__rtruediv__(Value(2))  即 3 / Value(2)
        def __rtruediv__(self, other):     # self=Value(2), other=3
            return other * self**-1
            #      │         │
            #      │         └─ Value(2).__pow__(-1)
            #      │            = Value(0.5, children=(Value(2),), local_grads=(-0.25,))
            #      │
            #      └─ 3 * Value(0.5)
            #         第1步：→ float.__mul__(Value) → NotImplemented
            #         第2步：→ Value(0.5).__rmul__(3)
            #                → self * other  →  Value(0.5).__mul__(3)
            #                  → other = Value(3)  # isinstance 检查，包装成 Value
            #                  → Value(1.5, children=(Value(0.5), Value(3)), local_grads=(3, 0.5))
        """
        return other * self**-1

    def __rmul__(self, other):
        """ other * self"""
        return self * other 
    
    def __rsub__(self, other):
        return other + (-self)

    def __radd__(self, other): 
        """ other + self"""
        return self + other  

    def log(self):
        """ log(self.data) """
        return Value(math.log(self.data), (self, ), (1/self.data, ))

    def exp(self):
        """ e^self.data"""
        return Value(math.exp(self.data), (self, ), (math.exp(self.data), ))
    
    def relu(self):
        """ max(0, self.data) """
        return Value(max(0, self.data), (self, ), (float(self.data > 0), ))  # f(x) = max(0, x)  ☞ little trick : float(True) = 1.0 VS float(False) = 0.0  
    

    def backward(self):
        """
        child.grad += v.grad * local_grad
        """
        topo = []
        visited = set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._children:
                    build_topo(child)
                topo.append(v)
        
        build_topo(self)
        self.grad = 1
        for v in reversed(topo):
            for child, local_grad in zip(v._children, v._local_grads):
                child.grad += local_grad * v.grad

# ④ Hyperparams + Parameter Initializaiton (state_dic) : store the knowledge of the model
# n_layer, n_embed, n_head : define the shape
# state_dict: wte, wpe, attn_wq/k/v/o, mlp_fc1/fc2, lm_head - all random Gaussian
# params = flat list[Value] of everything trainable

n_layer = 1
n_embd = 16
block_size = 16  # maximum context length of the attention window (note: the longest name is 15 characters)
n_head = 4 
head_dim =  n_embd // n_head

matrix = lambda nout, nin, std=0.08 : [ [Value(random.gauss(0, std)) for _ in range(nin)] for _ in range(nout)]  # nout怪不得，原来是外层数组也是矩阵的行
state_dic = {
    'wte': matrix(vocab_size, n_embd),
    'wpe': matrix(block_size, n_embd),
    'lm_head': matrix(vocab_size, n_embd)  # 转置发生在计算时，不在存储时；wte查第i行时embed,lm_head与每行做dot事unembed， 同一个数据布局服务两个方向。
}

for i in range(n_layer):
    state_dic[f'layer{i}.attn_wq'] = matrix(n_embd, n_embd)
    state_dic[f'layer{i}.attn_wk'] = matrix(n_embd, n_embd)
    state_dic[f'layer{i}.attn_wv'] = matrix(n_embd, n_embd)
    state_dic[f'layer{i}.attn_wo'] = matrix(n_embd, n_embd)
    
    state_dic[f'layer{i}.mlp_fc1'] = matrix(4 * n_embd, n_embd)  # 理解了矩阵的（输出矩阵， 输入矩阵），首先是它位置在 左侧 @ 
    state_dic[f'layer{i}.mlp_fc2'] = matrix(n_embd, 4 * n_embd)
    
params = [ p for mat in state_dic.values() for row in mat for p in row ]  # 列表推导式，从左到右 就是 从外循环到内循环
print(f"num params: {len(params)}")


# ⑤ Model Architecture: gpt(token_id, pos_id, keys, values) -> logits
# token_emb + pos_emb -> RMSNorm -> Nx(MHA + residual -> MLP -> residual) -> lm_head
# KV-cache passed explicitly (keys, values) for incremental inference

def linear(x, w):
    """
    我的语文、数学、英语成绩x [60, 80, 90]

    理科关注的比重不同w0: [0.8, 0.9, 0.5]
    文科关注的比重不同w1: [0.7, 0.6, 0.3]

    最后求一个综合得分： 
                    60 * 0.8 + 80 * 0.9 + 90 * 0.5 = 85分
                    60 * 0.7 + 80 * 0.6 + 90 * 0.3 = 56分
    """
    return [ sum(wi * xi for wi, xi in zip(wo, x)) for wo in w ]

def softmax(logits):
    """
                         e^(z_i)
    原始softmax(z_i) = ---------------
                       sum_j( e^(z_j) )


                                  e^(z_i)                 e^(z_i - max(z))
    数值稳定技巧版softmax(z_i) = ---------------       =  ----------------------------
                               sum_j( e^(z_j) )           sum_j( e^(z_j - max(z)) ) 
    
    """
    
    max_val = max(val.data for val in logits)          # 找logits 数组中最大值
    exps = [(val - max_val).exp() for val in logits]   # 求e的每一个logits - max_logit_value 次方
    total = sum(exps)                                  # 求次方的和作为分母
    return [e / total for e in exps]                   # 求每一个logit（数值稳定（-max_val）后）的百分比


def rmsnorm(x):
    """
                            x_i
    RMSNorm(x_i) =  ------------------------------   *  gamma_i
                    sqrt( 1/n * sum(x_j^2) + eps )

    """
    ms = sum(xi * xi for xi in x) / len(x)
    scale = (ms + 1e-5) ** -0.5 
    return [ xi * scale for xi in x ]


def gpt(token_id, pos_id, keys, values):
    tok_emb = state_dic['wte'][token_id] # vocab_size 传了个实际的 token_id 怎么就迷了呢？
    pos_emb = state_dic['wpe'][pos_id]
    x = [t + p for t, p in zip(tok_emb, pos_emb)]
    x = rmsnorm(x)  # karpathy’s design for backward pass via the residual connection

    for li in range(n_layer):
        # 1) Multi-head Attention block
        x_residual = x
        x = rmsnorm(x)

        q = linear(x, state_dic[f'layer{li}.attn_wq'])
        k = linear(x, state_dic[f'layer{li}.attn_wk'])
        v = linear(x, state_dic[f'layer{li}.attn_wv'])

        keys[li].append(k)
        values[li].append(v)

        x_attn = []  # 注意力是个列表？

        for h in range(n_head):
            hs = h * head_dim  # head start index

            q_h = q[hs : hs+head_dim]
            k_h = [ki[hs : hs+head_dim]  for ki in keys[li]]  # 历史keys 🔥 这就是为什么k-v cache这么重要？ 因为Attention注意力机制的计算公式：  softmax(q @ k.T / sqrt(d_k)) @ v  你并没看到其实 当前的q要跟所有历史位置的k算点积，从所有历史位置的v取加权和
            v_h = [vi[hs : hs+head_dim] for vi in values[li]]

            """
            attn_logits = []
            for t in range(len(k_h)):
                score = 0.0
                for j in range(head_dim):
                    score += q_h[j] * k_h[t][j]
                score = score / head_dim**0.5
                attn_logits.append(score)
            """
                        # 对第 t 个历史位置的 key 向量，跟当前query 做 dot product           除以 sqrt(head_dim) 做 scaling          取出每一个历史k的下标也就是位置
            # attn_logits = [ sum(q_h[j] * k_h[t][j] ) for j in range(head_dim)           / head_dim**0.5                        for t in range(len(k_h)) ]

            attn_logits = [ 
                sum(q_h[j] * k_h[t][j] for j in range(head_dim)) / head_dim ** 0.5  # ← 这个 for 属于 sum() 内部的生成器
                for t in range(len(k_h))                                            # ← 这个 for 才是列表推导的 for， 列表推导式的for 遵循左到右 对应展示式的上到下
            ]
            attn_weights = softmax(attn_logits)
            
            """
            head_out = []
            for j in range(head_dim):
                s = 0.0
                for t in range(len(v_h)):
                    s += attn_weights[t] * v_h[t][j]
                head_out.append(s)
            """
            head_out = [
                sum(attn_weights[t] * v_h[t][j] for t in range(len(v_h)))    # ← 这个 for 属于 sum() 内部的生成器
                for j in range(head_dim)                                     # ← 这个 for 才是列表推导的 for， 列表推导式的for 遵循左到右 对应展示式的上到下
            ]
            x_attn.extend(head_out)
        x = linear(x_attn, state_dic[f'layer{li}.attn_wo'])
        x = [a + b for a, b in zip(x, x_residual)]
        # 2) MLP Block 
        x_residual = x
        x = rmsnorm(x)

        x = linear(x, state_dic[f'layer{li}.mlp_fc1'])
        x = [xi.relu() for xi in x]
        x = linear(x, state_dic[f'layer{li}.mlp_fc2'])
        
        x = [a + b for a, b in zip(x, x_residual)]

    logits = linear(x, state_dic['lm_head'])
    return logits

# ⑥ Adam optimizer buffers : guide the parameter update:  Loss函数计算模型输出和目标之间的差距，然后反向传播得到每个参数的梯度 VS Adam用这些梯度结合自身的动量缓存（m, v）来更新参数，从而让loss变小
# m[i], v[i]: first/second moment estimates for each param, initialized to 0

"""
Adam 为了解决 w = w - lr * grad 中 lr 固定以及 grad 噪声大的问题，引入了：

动量平滑（
m：历史梯度的均值——"这个参数
一直往哪走"，用稳定方向代替单次噪声方向）
自适应步长（
v：历史梯度²的均值——"这个参数
一直抖多狠"，抖得狠就自动缩小步长，抖得少就放大步长）

Adam 不是调整全局学习率——它给每个参数装了一个独立的自适应油门。

"矩"就是"统计量"的学名：一阶矩 = 均值，二阶矩 = 均方值。没有更多含义。
"""
learning_rate, beta1, beta2, eps_adam = 0.01, 0.85, 0.99, 1e-8
m = [0.0] * len(params)
v = [0.0] * len(params)

# ⑦ Training loop: for each step
#       a) tokenize one doc: [BOS] + char_ids + [BOS]
#       b) forward: for each position, gpt() -> softmax -> cross-entropy loss
#       c) loss.backward(): fills p.grad for all params via autograd DAG
#       d) Adam update: bias-corrected m̂/v̂ → p.data -= lr * m̂/(√v̂ + ε)
#       e) zero gradients: p.grad = 0
num_steps = 1000  # number of training steps
for step in range(num_steps):

    # a) tokenize one doc: [BOS] + char_ids + [BOS]
    doc = docs[step % len(docs)] # 取余 其目的 是为了 num_step > voacb_size 的时候转回来 而不是 index out of range
    tokens = [BOS] + [ucharts.index(ch) for ch in doc] + [BOS]  # [1] + [2, 3, 7] + [9] = [1, 2, 3, 7, 9]  其实就是token id
    n = min(block_size, len(tokens) - 1)  # karpathy实战经验多，这是两者相比取实际，最长也就是block_size 也就是最长的名字也就是15个字符  vs 其他少的就用少的


    # b) forward: for each position, gpt() -> softmax -> cross-entropy loss
    keys, values = [ [] for _ in range(n_layer)], [ [] for _ in range(n_layer)]  # 一个 transformer block 一个自己的k-v cache list
    losses = []  
    for pos_id in range(n):
        token_id, target_id = tokens[pos_id], tokens[pos_id + 1]  # 当前词 与下一个词
        logits = gpt(token_id, pos_id, keys, values)
        probs = softmax(logits)
        loss_t = -probs[target_id].log()  # 交叉熵损失函数
        losses.append(loss_t)
    loss = (1/n) * sum(losses)  # final average loss over the document sequence 其实就是 计算模型生成一个完整名字（从BOS到BOS）的平均损失

    #  c) loss.backward(): fills p.grad for all params via autograd DAG
    loss.backward()

    # d) Adam update: bias-corrected m̂/v̂ → p.data -= lr * m̂/(√v̂ + ε)
    lr_t = learning_rate * (1 - step / num_steps)  # linear learning rate decay
    for i, p in enumerate(params): # faltten 的作用来了
        m[i] = beta1 * m[i] + (1 - beta1) * p.grad  # 等价于 m[i] = 前一次的m[i] * beta1 + 本次新梯度 * (1 - beta1)
        v[i] = beta2 * v[i] + (1 - beta2) * p.grad ** 2

        m_hat = m[i] / (1 - beta1 ** (step + 1))
        v_hat = v[i] / (1 - beta2 ** (step + 1))

        p.data -= lr_t * m_hat / (v_hat ** 0.5 + eps_adam)

        # e) zero gradients: p.grad = 0
        p.grad = 0  # 参数update完就立马清除历史梯度

    # .4f 表示将数字格式化为小数点后4位的浮点数
    # 4d 表示用4个字符宽度右对齐显示整数，不够位数会在左侧补空格
    print(f"step {step + 1 : 4d} / {num_steps : 4d} | loss {loss.data:.4f}", end='\r')


# ⑧ Inference: sample autoregressively
# start form BOS, gpt() -> softmax(logits/temperature)
# -> random.choices weighted by probs -> next token
# stop on BOS, decode token ids back to characters

temperature = 0.5
print("\n--- inference (new, hallucinated names) ---")  # karpathy用"hallucinated"表示这些名字是模型凭空生成（想象/创造）的，不一定基于训练集，体现LLM生成内容时“幻觉”现象的幽默说法。
for sample_idx in range(20):
    keys, values = [[] for _ in range(n_layer)], [[] for _ in range(n_layer)]  # TODO: 为什么推理的时候也有k-v cahe呢?  🔥🔥🔥 因为推理要做注意力 -> 注意力要求Q能看到 “所有历史” 的K,V -> 但每次只喂1个token -> 上一步算的K，V会随函数返回而消失 -> 所以需要外部容器把他们存住 ：= 这个list 就是那个 “外部容器”
                                                                               #                                          换句话说：KV cache不是因为 “要做注意力”， 而是因为 “一个一个喂↓， 但注意力要看全部历史” 这个矛盾。如果你一次把整个序列塞进去，注意力一样要做，但不需要cache
    token_id = BOS  # 分割符数字
    sample = []  # 存生成的一个名字

    for pos_id in range(block_size): # TODO: 为什么不是n 而是 block_size呢？  因为训练时你有答案，推理时你没答案。 训练过程：名字已知，比如 "Ana" → tokens = [BOS, A, n, a, BOS]  -> n = min(block_size, len(tokens) - 1)   # n = 4，因为序列就这么长  -> 循环4次就够了，多循环没有 target 可以算 loss      vs    推理：你不知道模型会生成多长的名字  ->  给满最大长度   ->   模型自己决定什么时候停 

        logits = gpt(token_id, pos_id, keys, values)   # TODO: 训练和推理同一套注意力机制？   训练：logits -> loss -> loss.backward() -> Adam 更新参数       vs     推理：logits -> softmax(logits / temp) -> sample采样 -> 下一个token
                    #  👆🏻 一个整数！ 不是序列
        probs = softmax([l / temperature for l in logits])
        token_id = random.choices(range(vocab_size), weights = [p.data for p in probs])[0]
        if token_id == BOS:
            break
        sample.append(ucharts[token_id])

    print(f"sample {sample_idx+1:2d} : {''.join(sample)}")




