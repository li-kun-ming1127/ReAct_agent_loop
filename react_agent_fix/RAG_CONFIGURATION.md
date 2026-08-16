# 私有 PDF RAG 运行配置

## 默认数据源

服务读取以下三份 PDF：

- `D:\pycharm\langchain\data\第一回：Matplotlib初识.pdf`
- `D:\pycharm\langchain\data\第二回：艺术画笔见乾坤.pdf`
- `D:\pycharm\langchain\data\第三回：布局格式定方圆.pdf`

修改 PDF 路径、文件内容、模型名或索引版本后，服务会根据新指纹创建新的 collection 和 Docstore，
不会删除旧索引。

## 默认模型

- 嵌入：本地 `D:\pycharm\langchain\models\text2vec-base-chinese`
- 文本/表格摘要：`deepseek-v4-flash`
- 图片：默认使用本地 Tesseract OCR 回退

DeepSeek 官方 V4 API 是纯文本模型。不要把 `deepseek-vl` 填入 DeepSeek 官方 API。若要结构化处理扫描图片和
复杂表格，需要配置一个真正支持图片输入、并兼容 OpenAI Chat Completions 的视觉服务：

```powershell
$env:VISION_MODEL="你的视觉模型名"
$env:VISION_MODEL_BASE_URL="视觉服务的 OpenAI-compatible 地址"
$env:VISION_API_KEY="视觉服务密钥"
```

完全本地构建（不把 PDF 内容发送给外部模型）：

```powershell
$env:ENABLE_MODEL_ENRICHMENT="0"
python mcp_server.py
```

此模式仍可 OCR、切分、建立 Multi-Vector 索引，但文本摘要使用截断回退，扫描表格不会恢复单元格结构。

## 启动顺序

终端一：

```powershell
cd D:\pycharm\langchain\LangChain_project
python mcp_server.py
```

首次启动会先构建索引，看到“Multi-Vector 索引构建完成”和“MCP 服务启动”后再启动客户端。

终端二：

```powershell
cd D:\pycharm\langchain\LangChain_project
streamlit run ReAct_agent_loop_with_toolnode.py
```

## 数据与日志位置

- v2 索引：`docs\multivector\matplotlib_v2`
- 原始图片资产：`docs\multivector\matplotlib_v2\assets`
- 索引清单：`docs\multivector\matplotlib_v2\index_manifest.json`
- MCP 运行日志：`mcp_server.log`
- Streamlit/Agent 日志：`ReAct_agent_loop_with_toolnode.log`
- 代码修改日志：`react_agent_fix\CODE_CHANGE_LOG.md`
