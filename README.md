# Qwen3 Web Demo

🤖 Qwen3 大模型 Web 交互演示 - 基于 Flask 的可视化聊天界面

---

## 📖 项目简介

本项目是一个基于 Flask 的 Web 应用，用于演示和交互 Qwen3 大语言模型。提供美观的前端界面，支持实时对话、参数调节和响应可视化。

**核心功能**:
- 💬 实时对话交互
- 🎨 可视化响应展示
- ⚙️ 模型参数调节（温度、top-p、max_tokens 等）
- 🖼️ 支持多模态输入（文本 + 图像）
- 📊 对话历史管理

---

## ✨ 功能特性

- **流式输出**: 实时显示模型生成的内容
- **参数可调**: 温度、重复惩罚、生成长度等
- **对话管理**: 保存、加载、清空对话历史
- **响应可视化**: 思考过程、置信度展示
- **移动端适配**: 支持手机访问

---

## 🚀 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置模型

下载 Qwen3 模型或配置 API 密钥:

```bash
# 本地模型路径（可选）
export QWEN3_MODEL_PATH="/path/to/qwen3-model"

# 或使用 API
export API_KEY="your-api-key"
```

### 3. 启动服务

```bash
python qwen3_web_demo.py
```

访问 http://localhost:5000

---

## 📁 文件说明

| 文件 | 说明 |
|------|------|
| `qwen3_web_demo.py` | 主程序，Flask Web 服务 |
| `qwen3_visualization_ideas.md` | 可视化设计文档 |
| `requirements.txt` | Python 依赖 |

---

## 🎨 界面预览

- 简洁现代的聊天界面
- 支持 Markdown 渲染
- 代码高亮显示
- 响应式布局

---

## ⚙️ 配置选项

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `temperature` | 采样温度 | 0.7 |
| `top_p` | 核采样 | 0.9 |
| `max_tokens` | 最大生成长度 | 2048 |
| `repetition_penalty` | 重复惩罚 | 1.1 |

---

## 💻 系统要求

- **Python**: 3.8+
- **内存**: 4GB+ (本地模型需要更多)
- **GPU**: 可选，CUDA 支持可加速推理

---

## 📋 依赖列表

```
flask>=2.3.0
torch>=2.0.0
transformers>=4.35.0
```

---

## 📄 许可证

MIT License
