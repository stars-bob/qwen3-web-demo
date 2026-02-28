#!/usr/bin/env python3
"""
Qwen3-1.7B Web Demo - 流式推理可视化
展示 token-by-token 生成过程和采样方法
"""

import os
import sys
import json
import time
import threading
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Generator, Optional

import torch
from flask import Flask, render_template_string, request, jsonify, Response
from transformers import AutoTokenizer, AutoModelForCausalLM

# ============ 配置 ============
MODEL_PATH = os.path.expanduser("~/models/qwen3-1.7b")
DEVICE = "cpu"
MAX_NEW_TOKENS = 512

# ============ 全局模型实例 ============
model = None
tokenizer = None

# ============ 采样配置 ============
@dataclass
class SamplingConfig:
    """采样配置"""
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 40
    repetition_penalty: float = 1.0
    do_sample: bool = True
    
    def to_dict(self):
        return asdict(self)

# ============ 加载模型 ============
def load_model():
    """加载 Qwen3-1.7B 模型"""
    global model, tokenizer
    
    if model is not None:
        return
    
    print(f"🔄 正在加载模型: {MODEL_PATH}")
    start = time.time()
    
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH,
        trust_remote_code=True,
        padding_side="left"
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.float32,
        device_map="cpu",
        trust_remote_code=True
    )
    model.eval()
    
    elapsed = time.time() - start
    print(f"✅ 模型加载完成! 耗时: {elapsed:.1f}s")
    print(f"   参数量: {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B")

# ============ 流式生成 ============
def generate_stream(
    messages: list,
    sampling_config: SamplingConfig
) -> Generator[dict, None, None]:
    """
    流式生成，逐步返回每个 token
    
    Yields:
        {
            "type": "token|think|answer|stats|done",
            "token": str,
            "token_id": int,
            "logits_info": {...},
            "timing": {...}
        }
    """
    start_time = time.time()
    
    # 应用聊天模板
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    # 编码输入
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(DEVICE)
    input_length = input_ids.shape[1]
    
    yield {
        "type": "info",
        "prompt_length": input_length,
        "prompt_tokens": input_ids[0].tolist()
    }
    
    # 生成配置
    generation_config = {
        "max_new_tokens": MAX_NEW_TOKENS,
        "do_sample": sampling_config.do_sample,
        "temperature": sampling_config.temperature if sampling_config.do_sample else 1.0,
        "top_p": sampling_config.top_p if sampling_config.do_sample else 1.0,
        "top_k": sampling_config.top_k if sampling_config.do_sample else 50,
        "repetition_penalty": sampling_config.repetition_penalty,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    
    # 手动逐步生成以可视化过程
    generated_ids = input_ids.clone()
    token_times = []
    
    in_think_mode = False
    think_content = []
    answer_content = []
    
    for step in range(MAX_NEW_TOKENS):
        step_start = time.time()
        
        # 前向传播
        with torch.no_grad():
            outputs = model(generated_ids)
            next_token_logits = outputs.logits[:, -1, :]
        
        # 应用温度缩放
        if sampling_config.do_sample and sampling_config.temperature != 1.0:
            next_token_logits = next_token_logits / sampling_config.temperature
        
        # 计算概率分布
        probs = torch.softmax(next_token_logits, dim=-1)
        
        # 获取 Top-K 和 Top-P 候选
        top_k_probs, top_k_indices = torch.topk(probs, k=min(sampling_config.top_k, probs.size(-1)))
        top_k_probs = top_k_probs[0].cpu().numpy()
        top_k_indices = top_k_indices[0].cpu().numpy()
        
        # 采样
        if sampling_config.do_sample:
            # Top-P 过滤
            sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
            cumsum_probs = torch.cumsum(sorted_probs, dim=-1)
            
            # 移除超过 top_p 的 token
            sorted_indices_to_remove = cumsum_probs > sampling_config.top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = False
            
            indices_to_remove = sorted_indices_to_remove.scatter(
                -1, sorted_indices, sorted_indices_to_remove
            )
            next_token_logits[indices_to_remove] = float('-inf')
            
            # 重新计算概率并采样
            filtered_probs = torch.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(filtered_probs[0], num_samples=1)
        else:
            # 贪婪解码
            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
        
        next_token_id = next_token.item()
        token_text = tokenizer.decode([next_token_id], skip_special_tokens=False)
        
        step_time = time.time() - step_start
        token_times.append(step_time)
        
        # 检测 <think> 标签
        if "<think>" in token_text:
            in_think_mode = True
            yield {"type": "think_start"}
            continue
        elif "</think>" in token_text:
            in_think_mode = False
            yield {"type": "think_end", "content": "".join(think_content)}
            continue
        
        # 收集内容
        if in_think_mode:
            think_content.append(token_text)
            content_type = "think"
        else:
            answer_content.append(token_text)
            content_type = "answer"
        
        # 构建 logits 信息（仅 Top-5）
        top_5_probs, top_5_indices = torch.topk(probs[0], k=5)
        logits_info = {
            "top_tokens": [
                {
                    "token_id": idx.item(),
                    "token": tokenizer.decode([idx.item()]),
                    "prob": prob.item(),
                    "logit": next_token_logits[0, idx].item()
                }
                for idx, prob in zip(top_5_indices, top_5_probs)
            ],
            "selected": {
                "token_id": next_token_id,
                "token": token_text,
                "prob": probs[0, next_token_id].item()
            }
        }
        
        yield {
            "type": content_type,
            "token": token_text,
            "token_id": next_token_id,
            "logits_info": logits_info,
            "timing": {
                "step_time_ms": step_time * 1000,
                "avg_time_ms": sum(token_times) / len(token_times) * 1000
            }
        }
        
        # 更新输入
        generated_ids = torch.cat([generated_ids, next_token.unsqueeze(0)], dim=1)
        
        # 检查结束
        if next_token_id == tokenizer.eos_token_id:
            break
    
    total_time = time.time() - start_time
    generated_tokens = len(token_times)
    
    yield {
        "type": "stats",
        "total_time": total_time,
        "generated_tokens": generated_tokens,
        "tokens_per_second": generated_tokens / total_time if total_time > 0 else 0,
        "avg_token_time_ms": sum(token_times) / len(token_times) * 1000 if token_times else 0,
        "think_content": "".join(think_content),
        "answer_content": "".join(answer_content)
    }
    
    yield {"type": "done"}

# ============ Flask App ============
app = Flask(__name__)

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Qwen3-1.7B 推理可视化</title>
    <style>
        :root {
            --bg: #1a1a2e;
            --surface: #16213e;
            --surface-light: #0f3460;
            --accent: #e94560;
            --text: #eaeaea;
            --text-muted: #a0a0a0;
            --success: #00d9ff;
            --warning: #ffa500;
        }
        
        * { box-sizing: border-box; margin: 0; padding: 0; }
        
        body {
            font-family: 'Segoe UI', system-ui, sans-serif;
            background: var(--bg);
            color: var(--text);
            min-height: 100vh;
            padding: 20px;
        }
        
        .container {
            max-width: 1400px;
            margin: 0 auto;
        }
        
        h1 {
            text-align: center;
            margin-bottom: 30px;
            font-size: 2em;
        }
        
        h1 span { color: var(--accent); }
        
        .main-grid {
            display: grid;
            grid-template-columns: 350px 1fr;
            gap: 20px;
        }
        
        @media (max-width: 1000px) {
            .main-grid { grid-template-columns: 1fr; }
        }
        
        /* 控制面板 */
        .control-panel {
            background: var(--surface);
            border-radius: 12px;
            padding: 20px;
            height: fit-content;
        }
        
        .panel-title {
            font-size: 1.1em;
            margin-bottom: 15px;
            padding-bottom: 10px;
            border-bottom: 1px solid var(--surface-light);
        }
        
        .form-group {
            margin-bottom: 15px;
        }
        
        label {
            display: block;
            margin-bottom: 5px;
            font-size: 0.9em;
            color: var(--text-muted);
        }
        
        input[type="text"], textarea, select {
            width: 100%;
            padding: 10px 12px;
            background: var(--surface-light);
            border: 1px solid transparent;
            border-radius: 8px;
            color: var(--text);
            font-size: 14px;
            transition: border-color 0.2s;
        }
        
        input[type="text"]:focus, textarea:focus, select:focus {
            outline: none;
            border-color: var(--accent);
        }
        
        textarea {
            min-height: 120px;
            resize: vertical;
            font-family: inherit;
        }
        
        .slider-group {
            display: flex;
            align-items: center;
            gap: 10px;
        }
        
        input[type="range"] {
            flex: 1;
            height: 6px;
            -webkit-appearance: none;
            background: var(--surface-light);
            border-radius: 3px;
            outline: none;
        }
        
        input[type="range"]::-webkit-slider-thumb {
            -webkit-appearance: none;
            width: 18px;
            height: 18px;
            background: var(--accent);
            border-radius: 50%;
            cursor: pointer;
        }
        
        .slider-value {
            min-width: 50px;
            text-align: center;
            font-family: monospace;
            background: var(--surface-light);
            padding: 4px 8px;
            border-radius: 4px;
        }
        
        .checkbox-group {
            display: flex;
            align-items: center;
            gap: 10px;
        }
        
        input[type="checkbox"] {
            width: 20px;
            height: 20px;
            accent-color: var(--accent);
        }
        
        button {
            width: 100%;
            padding: 14px;
            background: linear-gradient(135deg, var(--accent), #ff6b6b);
            border: none;
            border-radius: 8px;
            color: white;
            font-size: 16px;
            font-weight: 600;
            cursor: pointer;
            transition: transform 0.2s, box-shadow 0.2s;
        }
        
        button:hover:not(:disabled) {
            transform: translateY(-2px);
            box-shadow: 0 8px 20px rgba(233, 69, 96, 0.3);
        }
        
        button:disabled {
            opacity: 0.5;
            cursor: not-allowed;
        }
        
        /* 输出区域 */
        .output-area {
            display: flex;
            flex-direction: column;
            gap: 15px;
        }
        
        .output-panel {
            background: var(--surface);
            border-radius: 12px;
            padding: 20px;
        }
        
        .output-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 15px;
            padding-bottom: 10px;
            border-bottom: 1px solid var(--surface-light);
        }
        
        .token-stream {
            font-family: 'JetBrains Mono', 'Fira Code', monospace;
            font-size: 14px;
            line-height: 1.8;
            min-height: 100px;
            max-height: 400px;
            overflow-y: auto;
            padding: 15px;
            background: var(--surface-light);
            border-radius: 8px;
        }
        
        .token {
            display: inline;
            padding: 2px 0;
            border-radius: 3px;
            transition: background 0.2s;
        }
        
        .token:hover {
            background: rgba(233, 69, 96, 0.3);
        }
        
        .token.think {
            color: var(--text-muted);
            font-style: italic;
        }
        
        .token.think-tag {
            color: var(--warning);
            font-weight: bold;
        }
        
        .token.answer {
            color: var(--success);
        }
        
        .token.newline {
            display: block;
        }
        
        /* Logits 可视化 */
        .logits-panel {
            background: var(--surface);
            border-radius: 12px;
            padding: 20px;
        }
        
        .logits-list {
            display: flex;
            flex-direction: column;
            gap: 8px;
        }
        
        .logit-item {
            display: flex;
            align-items: center;
            gap: 10px;
            padding: 8px 12px;
            background: var(--surface-light);
            border-radius: 6px;
            font-family: monospace;
            font-size: 13px;
        }
        
        .logit-item.selected {
            background: rgba(0, 217, 255, 0.2);
            border: 1px solid var(--success);
        }
        
        .logit-rank {
            width: 24px;
            height: 24px;
            display: flex;
            align-items: center;
            justify-content: center;
            background: var(--surface);
            border-radius: 4px;
            font-size: 12px;
            color: var(--text-muted);
        }
        
        .logit-rank.top1 { background: gold; color: #000; }
        .logit-rank.top2 { background: silver; color: #000; }
        .logit-rank.top3 { background: #cd7f32; color: #fff; }
        
        .logit-token {
            flex: 1;
            font-weight: 500;
        }
        
        .logit-prob {
            min-width: 70px;
            text-align: right;
            color: var(--accent);
        }
        
        .prob-bar {
            width: 100px;
            height: 8px;
            background: var(--surface);
            border-radius: 4px;
            overflow: hidden;
        }
        
        .prob-bar-fill {
            height: 100%;
            background: linear-gradient(90deg, var(--accent), var(--success));
            border-radius: 4px;
            transition: width 0.3s;
        }
        
        /* 统计 */
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 15px;
        }
        
        .stat-card {
            background: var(--surface-light);
            padding: 15px;
            border-radius: 8px;
            text-align: center;
        }
        
        .stat-value {
            font-size: 1.8em;
            font-weight: bold;
            color: var(--accent);
        }
        
        .stat-label {
            font-size: 0.85em;
            color: var(--text-muted);
            margin-top: 5px;
        }
        
        /* 动画 */
        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.5; }
        }
        
        .generating {
            animation: pulse 1s infinite;
        }
        
        .cursor {
            display: inline-block;
            width: 8px;
            height: 18px;
            background: var(--accent);
            margin-left: 4px;
            animation: blink 1s infinite;
        }
        
        @keyframes blink {
            0%, 50% { opacity: 1; }
            51%, 100% { opacity: 0; }
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🧠 <span>Qwen3-1.7B</span> 推理可视化</h1>
        
        <div class="main-grid">
            <!-- 控制面板 -->
            <div class="control-panel">
                <div class="panel-title">⚙️ 采样配置</div>
                
                <div class="form-group">
                    <label>系统提示词</label>
                    <select id="system-prompt">
                        <option value="assistant">通用助手</option>
                        <option value="coder">代码助手</option>
                        <option value="creative">创意写作</option>
                        <option value="custom">自定义</option>
                    </select>
                </div>
                
                <div class="form-group" id="custom-system-group" style="display:none">
                    <label>自定义系统提示</label>
                    <textarea id="custom-system" placeholder="输入系统提示词..."></textarea>
                </div>
                
                <div class="form-group">
                    <label>用户输入</label>
                    <textarea id="user-input" placeholder="输入你的问题..."></textarea>
                </div>
                
                <div class="form-group">
                    <label>温度 (Temperature): <span class="slider-value" id="temp-value">0.7</span></label>
                    <div class="slider-group">
                        <input type="range" id="temperature" min="0.1" max="2.0" step="0.1" value="0.7">
                    </div>
                </div>
                
                <div class="form-group">
                    <label>Top-P: <span class="slider-value" id="topp-value">0.9</span></label>
                    <div class="slider-group">
                        <input type="range" id="top-p" min="0.1" max="1.0" step="0.05" value="0.9">
                    </div>
                </div>
                
                <div class="form-group">
                    <label>Top-K: <span class="slider-value" id="topk-value">40</span></label>
                    <div class="slider-group">
                        <input type="range" id="top-k" min="1" max="100" step="1" value="40">
                    </div>
                </div>
                
                <div class="form-group">
                    <label>重复惩罚: <span class="slider-value" id="rep-value">1.0</span></label>
                    <div class="slider-group">
                        <input type="range" id="repetition" min="1.0" max="2.0" step="0.1" value="1.0">
                    </div>
                </div>
                
                <div class="form-group">
                    <div class="checkbox-group">
                        <input type="checkbox" id="do-sample" checked>
                        <label for="do-sample" style="margin:0">启用采样 (Do Sample)</label>
                    </div>
                </div>
                
                <button id="generate-btn" onclick="startGeneration()">
                    🚀 开始生成
                </button>
            </div>
            
            <!-- 输出区域 -->
            <div class="output-area">
                <!-- Token 流 -->
                <div class="output-panel">
                    <div class="output-header">
                        <span class="panel-title">📜 生成过程</span>
                        <span id="status">就绪</span>
                    </div>
                    <div class="token-stream" id="token-stream">
                        <span style="color:var(--text-muted)">点击"开始生成"查看推理过程...</span>
                    </div>
                </div>
                
                <!-- Logits 可视化 -->
                <div class="logits-panel">
                    <div class="output-header">
                        <span class="panel-title">📊 当前 Token 概率分布 (Top-5)</span>
                        <span id="current-step">Step: 0</span>
                    </div>
                    <div class="logits-list" id="logits-list">
                        <div style="color:var(--text-muted);text-align:center;padding:20px">
                            等待生成开始...
                        </div>
                    </div>
                </div>
                
                <!-- 统计 -->
                <div class="output-panel">
                    <div class="output-header">
                        <span class="panel-title">📈 统计信息</span>
                    </div>
                    <div class="stats-grid" id="stats-grid">
                        <div class="stat-card">
                            <div class="stat-value" id="stat-tokens">0</div>
                            <div class="stat-label">生成 Token 数</div>
                        </div>
                        <div class="stat-card">
                            <div class="stat-value" id="stat-tps">0</div>
                            <div class="stat-label">Token/秒</div>
                        </div>
                        <div class="stat-card">
                            <div class="stat-value" id="stat-time">0s</div>
                            <div class="stat-label">总耗时</div>
                        </div>
                        <div class="stat-card">
                            <div class="stat-value" id="stat-avg">0ms</div>
                            <div class="stat-label">平均耗时/Token</div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>
    
    <script>
        const systemPrompts = {
            assistant: "You are a helpful assistant.",
            coder: "You are a helpful coding assistant. Provide clear, well-commented code.",
            creative: "You are a creative writing assistant. Be imaginative and engaging.",
            custom: ""
        };
        
        // 滑块更新
        document.querySelectorAll('input[type="range"]').forEach(slider => {
            slider.addEventListener('input', () => {
                const valueId = slider.id + '-value';
                document.getElementById(valueId).textContent = slider.value;
            });
        });
        
        // 系统提示词选择
        document.getElementById('system-prompt').addEventListener('change', (e) => {
            const customGroup = document.getElementById('custom-system-group');
            customGroup.style.display = e.target.value === 'custom' ? 'block' : 'none';
        });
        
        let eventSource = null;
        
        async function startGeneration() {
            const btn = document.getElementById('generate-btn');
            const status = document.getElementById('status');
            const stream = document.getElementById('token-stream');
            const logitsList = document.getElementById('logits-list');
            
            btn.disabled = true;
            status.textContent = '生成中...';
            status.classList.add('generating');
            stream.innerHTML = '<span class="cursor"></span>';
            
            // 关闭之前的连接
            if (eventSource) {
                eventSource.close();
            }
            
            // 获取参数
            const systemType = document.getElementById('system-prompt').value;
            const systemContent = systemType === 'custom' 
                ? document.getElementById('custom-system').value 
                : systemPrompts[systemType];
            
            const params = new URLSearchParams({
                user_input: document.getElementById('user-input').value || '你好，请介绍一下自己',
                system_prompt: systemContent,
                temperature: document.getElementById('temperature').value,
                top_p: document.getElementById('top-p').value,
                top_k: document.getElementById('top-k').value,
                repetition_penalty: document.getElementById('repetition').value,
                do_sample: document.getElementById('do-sample').checked
            });
            
            let inThink = false;
            let stepCount = 0;
            
            eventSource = new EventSource(`/generate?${params}`);
            
            eventSource.onmessage = (event) => {
                const data = JSON.parse(event.data);
                
                switch(data.type) {
                    case 'info':
                        console.log('Prompt length:', data.prompt_length);
                        break;
                        
                    case 'think_start':
                        inThink = true;
                        stream.innerHTML += '<span class="token think-tag">&lt;think&gt;</span>';
                        break;
                        
                    case 'think_end':
                        inThink = false;
                        stream.innerHTML += '<span class="token think-tag">&lt;/think&gt;</span><br>';
                        break;
                        
                    case 'think':
                    case 'answer':
                        stepCount++;
                        document.getElementById('current-step').textContent = `Step: ${stepCount}`;
                        
                        const tokenText = data.token
                            .replace(/&/g, '&amp;')
                            .replace(/</g, '&lt;')
                            .replace(/>/g, '&gt;')
                            .replace(/\\n/g, '<br>')
                            .replace(/ /g, '&nbsp;');
                        
                        const tokenClass = data.type === 'think' ? 'token think' : 'token answer';
                        const tokenSpan = `<span class="${tokenClass}" title="ID: ${data.token_id}, Prob: ${(data.logits_info.selected.prob * 100).toFixed(2)}%">${tokenText}</span>`;
                        
                        // 移除光标，添加 token，再加回光标
                        const cursor = stream.querySelector('.cursor');
                        if (cursor) cursor.remove();
                        stream.innerHTML += tokenSpan;
                        stream.innerHTML += '<span class="cursor"></span>';
                        stream.scrollTop = stream.scrollHeight;
                        
                        // 更新 logits 可视化
                        updateLogits(data.logits_info);
                        break;
                        
                    case 'stats':
                        document.getElementById('stat-tokens').textContent = data.generated_tokens;
                        document.getElementById('stat-tps').textContent = data.tokens_per_second.toFixed(2);
                        document.getElementById('stat-time').textContent = data.total_time.toFixed(1) + 's';
                        document.getElementById('stat-avg').textContent = data.avg_token_time_ms.toFixed(0) + 'ms';
                        break;
                        
                    case 'done':
                        status.textContent = '完成';
                        status.classList.remove('generating');
                        btn.disabled = false;
                        eventSource.close();
                        
                        // 移除光标
                        const finalCursor = stream.querySelector('.cursor');
                        if (finalCursor) finalCursor.remove();
                        break;
                }
            };
            
            eventSource.onerror = (error) => {
                console.error('Error:', error);
                status.textContent = '错误';
                status.classList.remove('generating');
                btn.disabled = false;
                eventSource.close();
            };
        }
        
        function updateLogits(logitsInfo) {
            const container = document.getElementById('logits-list');
            container.innerHTML = '';
            
            logitsInfo.top_tokens.forEach((item, idx) => {
                const isSelected = item.token_id === logitsInfo.selected.token_id;
                const rankClass = idx === 0 ? 'top1' : idx === 1 ? 'top2' : idx === 2 ? 'top3' : '';
                
                const div = document.createElement('div');
                div.className = `logit-item ${isSelected ? 'selected' : ''}`;
                div.innerHTML = `
                    <span class="logit-rank ${rankClass}">${idx + 1}</span>
                    <span class="logit-token">${item.token.replace(/\\n/g, '\\n') || '(空)'}</span>
                    <div class="prob-bar">
                        <div class="prob-bar-fill" style="width: ${item.prob * 100}%"></div>
                    </div>
                    <span class="logit-prob">${(item.prob * 100).toFixed(1)}%</span>
                `;
                container.appendChild(div);
            });
        }
    </script>
</body>
</html>
"""

@app.route('/')
def index():
    """主页"""
    return render_template_string(HTML_TEMPLATE)

@app.route('/generate')
def generate():
    """SSE 流式生成接口"""
    
    # 解析参数
    user_input = request.args.get('user_input', '你好')
    system_prompt = request.args.get('system_prompt', 'You are a helpful assistant.')
    
    sampling_config = SamplingConfig(
        temperature=float(request.args.get('temperature', 0.7)),
        top_p=float(request.args.get('top_p', 0.9)),
        top_k=int(request.args.get('top_k', 40)),
        repetition_penalty=float(request.args.get('repetition_penalty', 1.0)),
        do_sample=request.args.get('do_sample', 'true').lower() == 'true'
    )
    
    # 构建消息
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_input}
    ]
    
    def event_stream():
        for data in generate_stream(messages, sampling_config):
            yield f"data: {json.dumps(data)}\n\n"
    
    return Response(
        event_stream(),
        mimetype='text/event-stream',
        headers={
            'Cache-Control': 'no-cache',
            'X-Accel-Buffering': 'no'
        }
    )

@app.route('/health')
def health():
    """健康检查"""
    return jsonify({
        "status": "ok",
        "model_loaded": model is not None,
        "model_path": MODEL_PATH
    })

if __name__ == '__main__':
    print("=" * 60)
    print("🚀 Qwen3-1.7B Web Demo")
    print("=" * 60)
    
    # 加载模型
    load_model()
    
    print("\n📱 打开浏览器访问: http://localhost:5000")
    print("\n按 Ctrl+C 停止服务\n")
    
    # 运行 Flask
    app.run(
        host='0.0.0.0',
        port=5000,
        threaded=True,
        debug=False
    )