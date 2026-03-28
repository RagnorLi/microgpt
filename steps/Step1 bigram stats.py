"""
STEP 1：纯统计二元模型（零机器学习）
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
问题驱动：我想生成像人名一样的字符串，怎么做？
答案：统计训练数据里"每个字符后面跟着什么字符"的频率，按频率随机采样。
核心洞察：语言生成 = 从条件概率分布中采样
 
你将学到：
  - 分词（字符 → 整数）
  - 条件概率表 P(next | current)
  - 自回归采样（每次预测一步，把预测结果作为下一步输入）
 
遗留问题（驱动你进入Step 2）：
  ❌ 只能看1个字符的上下文，"em" 和 "am" 后面接什么，模型认为一样
  ❌ 纯记忆，无法泛化：训练集里没出现的二元组概率为0
  ❌ 不能"学习"，无法通过优化改进，只是计数
"""

import random
random.seed(42)

# ---数据-----
docs = [line.strip() for line in open('input.txt') if line.strip()]  # 一行名字作为一个列表中的元素
random.shuffle(docs)  # 打乱

uchars = sorted(set(''.join(docs)))  # ''用空连接list中every element
BOS = len(uchars)  # 【名字生成任务】 特殊token BOS=EOS: 句子开始/结束 ☞ 区分边界即可故可合并为1 ☞ 边界符 → 后验概率最高的是 姓氏。名字最后一个字 → 后验概率最高的是 边界符。 ＝ 🔄
                   # 【翻译/对话】BOS、EOS : 区分开始 + 结束 ☞ EOS (End of String) 是信号灯：告诉 Encoder，“输入结束了，请开始把你积累的知识压缩成一个向量”。  → BOS (Beginning of String) 是起搏器：告诉 Decoder，“现在开始吐第一个词，别管输入是什么，先给我个起始点”。 =  ➡️
vocab_size = len(uchars) + 1

print(f"文档数: {len(docs)}, 词汇表大小: {vocab_size}")

# ---统计二元组出现次数 -------
# counts[a][b] = “字符a后面跟字符b” 出现了多少次
counts = [[0] * vocab_size for _ in range(vocab_size)]  # [[0, ..., 0], ..., [0, ..., 0]]

for doc in docs:
    tokens = [BOS] + [uchars.index(ch) for ch in doc] + [BOS]
    for a, b in zip(tokens, tokens[1:]):  # 相邻对：（当前，下一个）
        counts[a][b] += 1

# --计数 -> 概率 （加1平滑，避免零概率）----
probs = []
for row in counts:
    total = sum(row) + vocab_size  # 拉普拉斯平滑：每格加1
    probs.append([c / total for c in row])

# --计算训练集上的平均损失（作为基准）---
import math
total_loss, total_n = 0, 0
for doc in docs[:1000]:
    tokens = [BOS] + [uchars.index(ch) for ch in doc] + [BOS]
    for a, b in zip(tokens, tokens[1:]):
        total_loss += -math.log(probs[a][b])
        total_n += 1
print(f"训练损失（二元统计模型）: {total_loss/total_n:.4f}")
# 注：随机猜测 loss ≈ ln(27) ≈ 3.30；越低越好 因为"完全不会"时，每次都等概率随机猜测下一个字符（含BOS一共27种），此时cross-entropy损失就是 -ln(1/27) = ln(27)


 # ── 自回归采样（生成名字）────────────────────────────────────
print("\n--- 生成的名字（纯统计）---")
for _ in range(10):
    token = BOS
    name = []
    for _ in range(20):   # 最多20个字符
        # 按概率分布随机选下一个token
        token = random.choices(range(vocab_size), weights=probs[token])[0]
        if token == BOS:  # 遇到BOS=句子结束
            break
        name.append(uchars[token])
    print(''.join(name))