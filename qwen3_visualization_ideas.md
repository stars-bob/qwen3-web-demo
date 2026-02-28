# Qwen3-1.7B 推理可视化实现思路

## 一、核心架构

```
┌─────────────────┐     SSE Stream      ┌─────────────────┐
│   Flask Backend │ ───────────────────► │   Frontend      │
│                 │   token-by-token    │   (EventSource) │
└────────┬────────┘                     └─────────────────┘
         │
    ┌────▼────┐
    │ Qwen3   │ ◄── 手动逐步生成 (非 generate() 方法)
    │ 1.7B    │     每个 forward 返回 logits
    └─────────┘
```

## 二、关键技术点

### 1. 流式生成 (Streaming Generation)

传统用法是一次性生成：
```python
# ❌ 传统方式 - 黑盒
outputs = model.generate(**inputs, max_new_tokens=100)
```

可视化需要**手动逐步生成**：
```python
# ✅ 可视化方式 - 白盒
for step in range(max_tokens):
    # 1. 前向传播
    outputs = model(input_ids)
    logits = outputs.logits[:, -1, :]  # 最后一个位置的 logits
    
    # 2. 应用采样策略 (可观测)
    probs = softmax(logits / temperature)
    next_token = sample(probs, top_p, top_k)
    
    # 3. 推送可视化数据
    yield {
        "token": next_token,
        "logits_info": get_top_k_tokens(logits, k=5),
        "timing": {...}
    }
    
    # 4. 更新输入，继续生成
    input_ids = concat(input_ids, next_token)
```

### 2. SSE (Server-Sent Events) 实时推送

```python
from flask import Response

def event_stream():
    for data in generate_stream(messages, config):
        yield f"data: {json.dumps(data)}\n\n"

return Response(event_stream(), mimetype='text/event-stream')
```

前端使用 `EventSource` 接收：
```javascript
const eventSource = new EventSource('/generate?...');
eventSource.onmessage = (event) => {
    const data = JSON.parse(event.data);
    updateUI(data);  // 实时更新
};
```

### 3. 采样过程可视化

每个 step 返回的数据结构：
```json
{
    "type": "answer",
    "token": "你",
    "token_id": 108264,
    "logits_info": {
        "top_tokens": [
            {"token": "你", "prob": 0.42, "logit": 12.5},
            {"token": "我", "prob": 0.31, "logit": 11.8},
            {"token": "这", "prob": 0.15, "logit": 10.2},
            {"token": "是", "prob": 0.08, "logit": 9.1},
            {"token": "我", "prob": 0.04, "logit": 8.3}
        ],
        "selected": {
            "token_id": 108264,
            "prob": 0.42
        }
    }
}
```

### 4. 思考过程分离

Qwen3 使用 `<think>` 标签标记内部思考：
```python
if "<think>" in token_text:
    in_think_mode = True
    yield {"type": "think_start"}
elif "</think>" in token_text:
    in_think_mode = False
    yield {"type": "think_end"}
else:
    yield {"type": "think" if in_think_mode else "answer", ...}
```

---

## 三、进阶可视化方案

### 方案 1: 注意力权重可视化

**原理**: 展示模型在生成当前 token 时，关注了输入的哪些位置。

```python
# 获取 attention weights
outputs = model(input_ids, output_attentions=True)
attentions = outputs.attentions  # 每层每头的注意力矩阵

# 可视化最后一层的注意力
last_layer_attn = attentions[-1][0]  # [num_heads, seq_len, seq_len]
```

**UI 效果**:
- 热力图展示注意力分布
- 点击某个生成的 token，高亮显示它关注的输入 token
- 支持切换不同层和注意力头

### 方案 2: 隐藏层状态 (Hidden States) 轨迹

**原理**: 将高维隐藏状态 (如 2048 维) 降维到 2D/3D，观察推理路径。

```python
outputs = model(input_ids, output_hidden_states=True)
hidden_states = outputs.hidden_states  # 每层的状态

# 使用 PCA/t-SNE/UMAP 降维
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
 trajectory = pca.fit_transform(hidden_states[-1][0].numpy())
```

**UI 效果**:
- 2D/3D 散点图展示 token 在语义空间中的位置
- 连线形成"推理轨迹"
- 相似语义的 token 聚集在一起

### 方案 3: KV Cache 可视化

**原理**: 展示 Key-Value 缓存的利用率，理解长上下文处理。

```python
past_kv = outputs.past_key_values  # 每层的 (key, value) 元组

# 分析 KV cache 大小
for layer_idx, (k, v) in enumerate(past_kv):
    print(f"Layer {layer_idx}: K shape={k.shape}, V shape={v.shape}")
```

**UI 效果**:
- 每层 KV cache 的内存占用可视化
- 显示哪些位置的 KV 被频繁访问
- 支持清除/压缩 KV cache 的操作

### 方案 4: 多种采样策略对比

**原理**: 并行使用不同采样参数，对比生成结果。

```python
configs = [
    {"temperature": 0.1, "name": "保守"},   # 贪婪近似
    {"temperature": 0.7, "name": "平衡"},   # 默认
    {"temperature": 1.5, "name": "创意"},   # 随机性强
]

# 同时生成三个版本
for config in configs:
    thread = threading.Thread(target=generate_with_config, args=(config,))
    thread.start()
```

**UI 效果**:
- 三列并排对比展示
- 每个版本显示温度和实际生成的差异

### 方案 5: 实时性能分析

**指标监控**:
```python
import time

step_times = []
memory_usage = []

for step in range(max_tokens):
    torch.cuda.synchronize()  # 确保 GPU 完成
    start = time.perf_counter()
    
    outputs = model(input_ids)
    
    torch.cuda.synchronize()
    step_time = time.perf_counter() - start
    step_times.append(step_time)
    
    # 内存监控
    if torch.cuda.is_available():
        mem = torch.cuda.memory_allocated() / 1024**2  # MB
        memory_usage.append(mem)
```

**可视化**:
- 折线图：每步耗时变化
- 直方图：token 生成时间分布
- 内存曲线：显存占用变化
- 瓶颈标记：指出哪一步变慢（可能是长上下文导致）

---

## 四、技术实现细节

### 1. 性能优化

**问题**: 逐 token 生成比原生 generate() 慢

**优化**:
```python
# 使用 torch.inference_mode() 代替 no_grad()
with torch.inference_mode():
    outputs = model(input_ids)

# 预热 (warmup) - 前几步较慢
for _ in range(3):
    _ = model(input_ids)

# 使用 past_key_values (KV Cache)
outputs = model(input_ids, past_key_values=past_kv)
past_kv = outputs.past_key_values  # 缓存供下一步使用
```

### 2. 前端优化

**虚拟滚动**: 大量 token 时避免 DOM 爆炸
```javascript
// 只渲染可见区域的 token
const VirtualList = ({ tokens }) => {
    const visibleTokens = tokens.slice(scrollTop / itemHeight, 
                                       (scrollTop + containerHeight) / itemHeight);
    return <div>{visibleTokens.map(...)}</div>;
};
```

**节流渲染**: 每 16ms (60fps) 最多渲染一次
```javascript
let pendingUpdate = null;
eventSource.onmessage = (event) => {
    pendingUpdate = JSON.parse(event.data);
    if (!animationFrameId) {
        animationFrameId = requestAnimationFrame(() => {
            render(pendingUpdate);
            animationFrameId = null;
        });
    }
};
```

### 3. 交互设计

**悬停查看详情**:
```javascript
<span class="token" 
      data-token-id={id}
      data-prob={prob}
      onMouseEnter={(e) => showTooltip(e, tokenDetails)}>
    {text}
</span>
```

**步骤回放**:
- 支持暂停生成
- 可以 "步进" 查看每一个 token 的生成过程
- 历史步骤可以回看

---

## 五、未来扩展

### 多模态可视化
- 对于多模态模型，可视化图像 token 的生成过程
- 展示文本-图像的跨模态注意力

### 对比可视化
- 同一个 prompt，不同模型的生成过程对比
- 微调前后模型的行为差异

### 可解释性增强
- 集成 LIME/SHAP，解释特定 token 的选择原因
- 展示激活的神经元及其对应的知识概念

### 协作分析
- 支持多人同时观察同一次生成
- 标注和评论特定步骤

---

## 六、总结

当前 Demo 实现了**基础可视化**：
- ✅ Token-by-token 流式展示
- ✅ 采样参数可调
- ✅ Top-5 概率分布
- ✅ 思考过程分离

进阶方向：
- 🎯 注意力权重热力图
- 🎯 隐藏层状态轨迹
- 🎯 多采样策略对比
- 🎯 实时性能分析

核心思想是**把黑盒变白盒**，让用户直观理解 LLM 是如何一步步"思考"和"创作"的。
