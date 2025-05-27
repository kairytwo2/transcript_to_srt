# 转录文件转换为 WebVTT

该仓库提供 `convert_to_vtt.py` 脚本，可将语音转文字得到的转录 JSON 文件转换为英文或中英双语的 WebVTT 字幕。

## 环境要求

- Python 3.10 及以上
- 依赖包见 `requirements.txt`

安装依赖：

```bash
pip install -r requirements.txt
```

## 使用方法

运行前先设置以下环境变量：

- `OPENAI_API_KEY` – 你的 OpenAI API key
- `OPENAI_API_BASE_URL` – API 基础地址（可选）
- `OPENAI_MODEL` – 用于翻译的模型名称

示例命令：

```bash
python convert_to_vtt.py transcript.json
```

可以使用以下参数：

- `--model MODEL_NAME` 指定模型
- `--concurrency N` 控制并发数量
- `--lang LANGUAGE` 额外翻译成该语言（若仅需英文可省略）
- `--font-size SIZE` 以 `em` 为单位调整字幕字体大小
- `--font-color COLOR` 设置字体颜色（默认 `white`）
- `--background-color COLOR` 设置背景颜色（默认 `transparent`）
- `--output PATH` 指定输出文件位置

若未提供 `--output`，脚本将在当前目录生成 `<transcript>.vtt` 文件。
