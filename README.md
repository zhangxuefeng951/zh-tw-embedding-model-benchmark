# 繁體中文 Embedding / Reranker 檢索評測

用繁體中文資料集評測 Embedding（文字向量）模型與 Reranker（重新排序）模型的檢索能力。
只要模型服務提供常見的 **`/v1/embeddings`** API，就可以在 Web UI 設定端點、模型名稱與**前綴（Prefix）**後直接評測。

* 使用繁體中文評測各家 Embedding 模型的檢索能力：https://ihower.tw/blog/archives/12167
* 使用繁體中文評測各家 Reranker 模型的重排能力：https://ihower.tw/blog/archives/12227

## 功能

- **Web UI**（Flask + HTML/CSS/JavaScript）與 **CLI**，兩者共用同一套核心
- **Embedding provider**
  - OpenAI 相容 `/v1/embeddings`：vLLM、TEI、Infinity、LiteRT-LM、Ollama、OpenAI、Mistral、Jina…
  - 原生 API：Cohere、Voyage、Google Gemini、Vertex AI
  - 本地載入（選裝）：sentence-transformers、transformers、FlagEmbedding、model2vec
- **前綴**
  - 查詢、文件各自設定；一次可以勾選多組「查詢前綴 × 文件前綴」比較
  - 文件前綴可以用 `{title}` 帶入文件標題
  - 即時預覽實際送出的文字，尾端空白與換行都看得見
- **內建預設檔**：E5 三個版本、EmbeddingGemma 2，以及舊版腳本用過的所有模型
- **資料集**
  - DRCD test／dev（GitHub 原始版，含標題）、TCEval-v2 drcd
  - C-MTEB 8 個檢索子集，各有簡體原文與 OpenCC 轉繁體兩種版本
  - 自訂上傳：簡單 CSV／JSONL，或 BEIR 格式
- **指標**
  - Hit@K、MRR@K、NDCG@K、Recall@K
  - 系統保存每題正解的完整名次，改 K 值不需要重跑
- **向量維度**
  - 可以送 `dimensions` 參數給 API
  - 也可以勾選多個維度，由系統截取前 N 維後重新正規化再評測，不必重新呼叫 API
- **Reranker 二階段評測**：`/v1/rerank`（Jina、Cohere、Voyage、vLLM、TEI）、LLM 列表式排序（RankGPT）、本地 CrossEncoder／BGE／Qwen3-Reranker
- **BM25 基準**：jieba 斷詞，計分公式與舊版的 `rank_bm25` 完全相同
- **報告**：排行榜、單一結果頁、K 值曲線、逐題錯誤分析、向量健康檢查（NaN／全零／長度），可匯出 CSV、JSON、Markdown
- **執行**
  - 背景任務佇列：預設一次跑 1 個任務，可以在設定改成 2 個以上
  - 進度條、可以取消
  - 向量分批存入快取：中斷後重跑會接續；只改查詢前綴時，文件向量也會直接沿用
- **舊版 `data.db` 匯入**：用新的指標重新計算舊結果，在排行榜標示「舊資料匯入」

## 快速開始

需要 Python 3.13 與 [uv](https://docs.astral.sh/uv/)。

```bash
uv sync                      # 安裝（只用 API 評測）
uv sync --extra local        # 需要由本系統直接載入本地模型時才安裝（torch 等）
uv run embedbench serve      # 開啟 http://127.0.0.1:5000
```

API Key 放在專案根目錄的 `.env`（已在 `.gitignore`）或系統環境變數，Web UI 只填「環境變數名稱」。從範本複製後，只填會用到的項目：

```bash
cp .env.example .env
```

| 項目 | 用途 |
|---|---|
| `OPENAI_API_KEY`、`COHERE_API_KEY`、`VOYAGE_API_KEY`、`GEMINI_API_KEY`、`MISTRAL_API_KEY`、`JINA_API_KEY` | 各家雲端 API |
| `GCP_PROJECT_ID`、`GCP_ACCESS_TOKEN` | Vertex AI |
| `LOCAL_API_KEY` | 自己啟動的服務設定了 `--api-key` 時使用（名稱可以自訂） |
| `HF_TOKEN`、`HF_ENDPOINT`、`HF_HOME` | 本地載入需要授權的模型（例如 `google/embeddinggemma-2`）、鏡像站、下載快取位置 |
| `EMBEDBENCH_DATA_DIR`、`EMBEDBENCH_PRESETS_DIR` | 資料目錄、預設檔資料夾 |

- 只評測自己啟動的本地 API 時，通常什麼都不用填
- 系統環境變數的優先順序高於 `.env`
- API Key 改了立即生效；其他設定修改後請重新啟動

### 1. 自行啟動模型 API 服務

本系統**不負責**啟動模型服務。以下是常見的做法，請依你的環境與版本調整：

```bash
# vLLM（OpenAI 相容，提供 /v1/embeddings；Base URL 填 http://localhost:8000/v1）
vllm serve intfloat/multilingual-e5-large --runner pooling

# Hugging Face TEI（Base URL 填 http://localhost:8080/v1）
docker run -p 8080:80 -v $PWD/hf:/data ghcr.io/huggingface/text-embeddings-inference:cpu-1.9 \
  --model-id intfloat/multilingual-e5-large
```

EmbeddingGemma 2 的推論工具版本、精度（不要用 float16）與部署注意事項，請見 [`docs/EmbeddingGemma 2 使用注意事項`](docs/EmbeddingGemma%202%20使用注意事項)。
**特別注意：如果服務端會自動幫文字加前綴，請把本系統的前綴清空，避免重複加前綴。**

### 2. 在 Web UI 評測

1. 「新增評測」→ 選預設檔（例如 `EmbeddingGemma 2`）→ 填 **API 端點**與**模型名稱**
2. 勾選要比較的**前綴組合**，在預覽區確認實際送出的文字
3. 勾選資料集；需要的話填「評測維度」（例如 `768,512,256,128`）或「快速試跑」題數
4. 按「**測試連線**」：系統送出一筆查詢與文件，回報向量維度、向量長度、有沒有 NaN
5. 按「開始評測」，到「任務」頁看進度，完成後到「排行榜」比較結果

### 3. 或用 CLI

```bash
uv run embedbench presets -v                     # 列出預設檔與前綴組合
uv run embedbench test --preset embeddinggemma-2 --base-url http://localhost:8000/v1
uv run embedbench run  --preset embeddinggemma-2 --base-url http://localhost:8000/v1 \
    --dataset drcd-test --all-pairs --dims 768,512,256,128
uv run embedbench run  --preset multilingual-e5-large --base-url http://localhost:8000/v1 --limit 200
uv run embedbench run  --base-url http://localhost:8000/v1 --model my-model \
    --query-prefix "query: " --document-prefix "passage: "
uv run embedbench bm25 --dataset drcd-test
uv run embedbench rerank --source <結果 ID> --preset cohere-rerank-v3.5 --candidates 100
uv run embedbench results --dataset drcd-test --k 1,3,5,10 --format md
uv run embedbench run --config my-run.yaml       # 設定檔欄位與 Web API 相同
uv run embedbench --help
```

`my-run.yaml` 範例：

```yaml
provider: openai
params:
  base_url: http://localhost:8000/v1
  model: google/embeddinggemma-2
prefix_pairs:
  - label: 搜尋文件
    query: "task: search result | query: "
    document: "title: none | text: "
datasets: [drcd-test, drcd-dev]
eval_dims: [768, 256]
```

## 前綴（Prefix）

很多 embedding 模型在訓練時，就學到「查詢」和「文件」要用不同的開頭標記。前綴不只是裝飾，而是模型學到的輸入格式；用錯或省略前綴都可能讓檢索變差。

本系統把前綴當成評測設定的一部分：

- 每組前綴各自產生一個結果，可以在排行榜直接比較
- 同一個模型改了前綴就會重新計算向量；不同前綴的向量不會混用

### E5

| 模型 | 查詢前綴 | 文件前綴 |
|---|---|---|
| `intfloat/multilingual-e5-large`、`intfloat/e5-large-v2` | `query: ` | `passage: ` |
| `intfloat/multilingual-e5-large-instruct` | `Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery: ` | （不加） |

### EmbeddingGemma 2

| 用途 | 前綴 |
|---|---|
| 搜尋文件 | `task: search result \| query: ` |
| 問答系統 | `task: question answering \| query: ` |
| 文字分類 | `task: classification \| query: ` |
| 文字相似度 | `task: sentence similarity \| query: ` |
| 文件內容 | `title: none \| text: `（有標題時可用 `title: {title} \| text: `） |

（表格中的 `\|` 是 Markdown 跳脫，實際前綴是單一直線 `|`。）

### 前綴規則

- 前綴尾端的空白也算在前綴內；Web UI 預覽以 `␣` 表示空白、以 `↵` 表示換行
- 文件前綴可以使用 `{title}`：會代入文件標題，沒有標題時代入 `none`
- 可以使用 `{text}` 指定原文的位置；沒有寫 `{text}` 時，原文直接接在前綴後面
- Cohere、Voyage、Gemini、Vertex 的原生 API 用 `input_type`／`task_type` 參數區分查詢與文件，在 provider 參數中設定，通常不需要文字前綴

## 資料集

| key | 說明 |
|---|---|
| `drcd-test` | DRCD 測試集（3493 題、1000 段落，附標題），來源：[DRCD GitHub](https://github.com/DRCKnowledgeTeam/DRCD)，CC BY-SA 3.0 |
| `drcd-dev` | DRCD 開發集（3524 題、1000 段落，附標題） |
| `tceval-drcd-test` | 舊版腳本使用的 [TCEval-v2](https://huggingface.co/datasets/MediaTek-Research/TCEval-v2) drcd（題目同 DRCD test，沒有標題） |
| `cmteb-<子集>` | C-MTEB 檢索資料集簡體原文：T2、MMarco、Du、Covid、Cmedqa、Medical、Ecom、Video |
| `cmteb-<子集>-tw` | 同上，以 OpenCC `s2twp`（含台灣用詞）轉成繁體 |

內建資料集第一次使用時會自動下載並轉換，存放在 `data/datasets/`。DRCD 從 GitHub 下載；TCEval 與 C-MTEB 從 Hugging Face 下載。C-MTEB 每個子集的文件庫約 10 萬筆，評測成本較高。

### 自訂資料集

在「資料集」頁上傳，或用 CLI 匯入：

- **簡單格式**（CSV／TSV／JSONL）
  - 每列一題，欄位：`question`、`paragraph`（必填）、`title`（選填）
  - 相同段落會自動合併成同一篇文件
- **BEIR 格式**
  - 三個檔案：`corpus.jsonl`（`_id`、`title`、`text`）、`queries.jsonl`（`_id`、`text`）、`qrels.tsv`（`query-id`、`corpus-id`、`score`）
  - 也可以上傳 zip
  - 支援一題多個正解與正解分級

```bash
uv run embedbench datasets import-simple my.csv --name 我的資料
uv run embedbench datasets import-beir --zip my-beir.zip --name 我的 BEIR
```

## 指標

一律先把向量正規化成長度 1，再用 cosine 相似度在整個文件庫中排序。

| 指標 | 定義 |
|---|---|
| Hit@K | 前 K 名內有任何正解的比例（舊版 `benchmark.py` 的 hit rate，舊版固定 K=5） |
| MRR@K | 第一個正解名次的倒數，排在 K 名外記 0（舊版的 MRR，K=5） |
| NDCG@K | 考慮正解分級與名次的折扣累積增益 |
| Recall@K | 前 K 名內找回的正解比例；每題只有一個正解時等於 Hit@K |

## 新增模型

- **只是換一個 API 模型**：在 Web UI 選「自訂：任何 /v1/embeddings 服務」，填端點、模型與前綴；調好之後可以「另存為自訂預設檔」
- **加入內建預設檔**：編輯 [`presets/embedding.yaml`](presets/embedding.yaml)（Reranker 是 [`presets/reranker.yaml`](presets/reranker.yaml)），重新整理頁面就會出現。也可以在「預設檔」頁把自訂預設檔匯出成 YAML 貼進去
- **新的 API 類型**：在 `embedbench/providers/` 繼承 `EmbeddingProvider`，定義 `params` 並實作 `embed_batch()`，再加進 `providers/__init__.py`；Web UI 的表單會依 `params` 自動產生

## 匯入舊版 data.db

舊版 `index-*.py` 把向量存在 SQLite `data.db`，可以匯入新系統：

- **匯入方式**：在「設定」頁填路徑，或執行 `uv run embedbench import-legacy data.db`
- **對應方式**：段落以文字比對；問題以 `dataset_id`（TCEval）或文字比對
- **匯入結果**：用新的指標重新計算，在排行榜標示「舊資料匯入」（前綴等設定沒有記錄）
- **限制**：早期使用 Supabase 時建立的向量不在 `data.db` 裡，無法匯入

## 舊腳本對照

舊腳本已經改寫進新架構並移除（可以從 git 歷史找回）：

| 舊腳本 | 新架構 |
|---|---|
| `migrate.py`、`benchmark.py` | 自動建立資料庫；評測與指標在 `embedbench/ranking.py` |
| `index-openai.py`、`index-mistral.py` | provider `openai`，預設檔 `openai-text-embedding-3-small`、`mistral-embed` |
| `index-cohere.py`／`index-voyage*.py`／`index-gemini.py`／`index-vertex.py` | provider `cohere`／`voyage`／`gemini`／`vertex` |
| `index-sentence-transformers.py`、`index-embeddinggemma.py`、`index-nomic.py`、`index-qwen3.py`、`index-nvidia.py` | provider `sentence_transformers`，每個模型都有預設檔 |
| `index-chuxin.py`／`index-model2vec.py`／`index-ckiplab.py`、`test-ckiplab.py` | provider `flag_embedding`／`model2vec`／`transformers`（CLS pooling） |
| `index-twcc.py` | 不再支援（台智雲 FFM 專用 API） |
| `rank-bm25.py` | BM25 基準（`embedbench bm25`） |
| `reranker-jina.py`、`reranker-cohere.py`、`reranker-voyage*.py` | reranker `rerank_api` |
| `reranker-gpt.py` | reranker `llm_listwise` |
| `reranker-bge.py`、`reranker-cross-encoder.py`、`reranker-qwen.py` | reranker `flag_reranker`、`cross_encoder`、`qwen3_reranker` |

`chunking/` 是另一個獨立實驗（切分策略評測），需要 `chunking_evaluation` 套件與另外準備的語料檔（`data/*.txt`，不在 repo 中），不屬於本系統。

## 專案結構

```
embedbench/
  cli.py            命令列
  web/              Flask Web UI（api.py、templates/、static/）
  providers/        Embedding provider（api.py、local.py）
  rerankers/        Reranker（api.py、local.py）
  datasets/         資料集（builtin.py、custom.py）
  runner.py         評測流程（embedding、BM25、reranker、連線測試）
  jobs.py           背景任務佇列
  ranking.py        檢索排名與指標
  vectors.py        可續跑的向量快取
  bm25.py           BM25（scipy 稀疏矩陣實作）
  legacy.py         舊版 data.db 匯入
presets/            內建預設檔（YAML）
tests/              pytest（含模擬 /v1/embeddings 的測試伺服器）
data/               執行期資料（資料庫、資料集、向量快取、結果），不進 git
```

## 開發

```bash
uv run pytest
uv run python tests/mock_server.py --port 8765   # 模擬的 /v1/embeddings，可以用來試操作 Web UI
```

Web UI 預設只監聽 `127.0.0.1`，而且沒有登入機制；若要用 `--host 0.0.0.0` 開放到其他電腦，請先自行評估風險。
