# 锵锵三人行字幕生成系统

🎬 自动下载《锵锵三人行》节目并生成中文字幕的 Python 工具集

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 📖 项目简介

本项目用于从 YouTube 下载《锵锵三人行》节目视频，使用 OpenAI Whisper 或其他 ASR 引擎自动生成中文字幕（SRT 格式）。

**核心功能**:
- 🔍 自动搜索指定年份的节目视频
- 🧹 智能去重，按日期保留最佳版本
- ⬇️ 下载音频（仅用于转录，节省空间）
- 🎙️ ASR 转录生成 SRT 字幕
- 📊 处理进度和统计报告

---

## ✨ 功能特性

- **多 ASR 后端支持**: 本地 Whisper / Faster-Whisper / Groq API
- **智能去重**: 按播出日期去重，优先选择高清完整版
- **断点续传**: 支持中断后继续处理
- **日志记录**: 完整的处理日志
- **CPU 优化**: 支持 CPU 推理，无需 GPU

---

## 🚀 快速开始

### 1. 安装依赖

```bash
# 克隆项目
git clone <repository-url>
cd qiangqiang-asr

# 安装 Python 依赖
pip install -r requirements.txt

# 安装系统依赖（Ubuntu/Debian）
sudo apt-get update
sudo apt-get install -y ffmpeg

# 安装 yt-dlp
pip install -U yt-dlp
```

### 2. 配置环境变量（可选）

如果使用 Groq API 版本:

```bash
export GROQ_API_KEY='gsk_your_api_key_here'
```

### 3. 运行

```bash
# 使用重构版（推荐）
python qiangqiang_cpu_refactored.py 2016

# 限制处理数量（测试用）
python qiangqiang_cpu_refactored.py 2016 --limit 5

# 更换模型
python qiangqiang_cpu_refactored.py 2016 --model large-v3

# 使用 Groq API
python qiangqiang_groq.py 2016
```

---

## 📁 文件说明

| 文件 | 说明 | 推荐度 |
|------|------|--------|
| `qiangqiang_cpu_refactored.py` | **推荐主版本**，面向对象设计，类型注解完整 | ⭐⭐⭐⭐⭐ |
| `qiangqiang_groq.py` | Groq API 版本，云端快速处理 | ⭐⭐⭐⭐ |
| `qiangqiang_pipeline.py` | 生产级 Whisper 版本 | ⭐⭐⭐⭐ |
| `qiangqiang_final.py` | 多后端支持的流水线 | ⭐⭐⭐ |

---

## ⚙️ 配置选项

### Whisper 模型选择

| 模型 | 准确率 | 速度 | 推荐场景 |
|------|--------|------|----------|
| `tiny` | 低 | 极快 | 测试 |
| `base` | 中 | 快 | 快速预览 |
| `small` | 中高 | 中等 | 平衡选择 |
| `medium` | 高 | 较慢 | 推荐 |
| `large-v3` | 最高 | 慢 | 最终版本 |

### 常用参数

```bash
# 处理指定年份
python qiangqiang_cpu_refactored.py 2016

# 限制处理数量
python qiangqiang_cpu_refactored.py 2016 --limit 10

# 选择模型
python qiangqiang_cpu_refactored.py 2016 --model medium

# 只列出视频，不下载
python qiangqiang_pipeline.py 2016 --list-only
```

---

## 📂 输出结构

```
qiangqiang_subtitles/
├── 2016/
│   ├── metadata.json          # 视频元数据
│   ├── 20160101_锵锵三人行.srt
│   ├── 20160102_锵锵三人行.srt
│   └── ...
├── 2017/
│   └── ...
└── download.log               # 处理日志
```

---

## 💻 系统要求

- **Python**: 3.8 或更高
- **操作系统**: Linux / macOS / Windows (WSL 推荐)
- **内存**: 4GB+ (推荐 8GB)
- **磁盘空间**: 10GB+ (用于临时音频文件)
- **网络**: 可访问 YouTube

### 可选硬件加速

- **NVIDIA GPU**: 使用 CUDA 加速 Whisper 推理
- **Apple Silicon**: M1/M2/M3 芯片支持

---

## 🔧 常见问题

### Q: 下载速度很慢？

A: YouTube 下载速度受网络环境影响。可以尝试:
- 使用代理
- 降低音频质量 (`--audio-quality 48K`)

### Q: 转录准确率不高？

A: 尝试更换更大的模型:
```bash
python qiangqiang_cpu_refactored.py 2016 --model large-v3
```

### Q: 如何处理特定月份？

A: 目前不支持直接按月份筛选，可以:
1. 运行 `--list-only` 查看视频列表
2. 编辑生成的 `metadata.json` 删除不需要的条目
3. 重新运行处理

### Q: 遇到 "Video unavailable" 错误？

A: 该视频可能在您所在地区不可用或被删除。脚本会自动跳过并继续处理其他视频。

---

## 📋 依赖列表

```
faster-whisper>=0.10.0
yt-dlp>=2023.0.0
requests>=2.28.0
```

系统依赖:
- `ffmpeg` (必需)
- `python3-dev` (编译需要)

---

## ⚠️ 免责声明

1. **版权问题**: 《锵锵三人行》节目版权归原制作方所有。本项目仅供个人学习和研究使用，请勿用于商业用途或公开传播。
2. **YouTube 条款**: 使用本工具需遵守 YouTube 服务条款。
3. **ASR 准确性**: 自动生成的字幕可能存在错误，建议人工校对后使用。

---

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！

---

## 📄 许可证

本项目采用 MIT 许可证。详见 [LICENSE](LICENSE) 文件。

---

## 🙏 致谢

- [OpenAI Whisper](https://github.com/openai/whisper) - ASR 引擎
- [Faster-Whisper](https://github.com/SYSTRAN/faster-whisper) - 优化的 Whisper 实现
- [yt-dlp](https://github.com/yt-dlp/yt-dlp) - YouTube 下载工具
- [Groq](https://groq.com/) - 云端 AI 推理平台

---

**注意**: 本项目为个人学习项目，与《锵锵三人行》制作方无关联。
