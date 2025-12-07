# LLM Extractor & RAG App


**Tech Stack:** 
- LangGraph/LangChain
- Chroma vector store (in-memory) – interchangeable with your own providers.
- OpenAI models – interchangeable with your own providers.

## Index-Process

### OCR-Indexer: OCR-Extraction with llm metadata enrichment

![OCR-Indexer](/docs/index_ocr_graph.png)

### LLM-Indexer: LLM-Extraction and Splitting with metadata enrichment

![LLM-Indexer](/docs/index_ocr_graph.png)

## Retrieval-Process: 2-Stage with LLM-Compression

![Retrieval-Graph](/docs/index_ocr_graph.png)



## Setup & Launch

1. **Prerequisites:**  
   - Install *poppler* for `pdf2image`, and adjust the path in `.env` if necessary:  
     - https://github.com/oschwartz10612/poppler-windows/releases/  
     - Define the path in `.env` as `POPLER_PATH`
   - Add environment variables like `OPENAI_API_KEY` in `.env`
   - Optional for LangGraph/LangSmith UI: `LANGSMITH_API_KEY`

2. **Install dependencies & start the local runtime:**

```bash
python -m venv .venv          # alternatively: py -3.12 -m venv .venv
source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -e .[dev]
langgraph dev 
```

## How to Use the Graphs

### LangSmith UI

1. **Select a graph:** After running `langgraph dev`, you can switch between `index_graph` and `retrieval_graph` via the LangGraph UI (or LangSmith, if connected).
2. **Set parameters:** Each graph accepts structured inputs. Before execution, ensure that `collection_id`, `doc_id`, and (if applicable) `chunking_config` match your documents.

**Retrieval Graph**
Example with an already indexed collection in `path: ./data/DB_ZB25.pdf`:

* `collection_id` = DB_ZB, `doc_id` = DB_ZB_S55 (95776bb8-7907-486c-b1fb-04e456c44e2c)
* `query`:
  * Was macht die DB in Bezug auf Nachhaltigkeit?
  * Wie viele Arbeitskräfte hat der DB Konzern auf der ganzen Welt?

**Index Graph**
Example Run 1:

* `collection_id: Test_M`
* `doc_id: Test_M_1`
* `path: ./data/Test_M1/Test_M_1.pdf`



Once indexed, you can query both the collection and individual documents. You can also copy your own PDFs into `./data` and index them.

### Tests

* The same use cases can be executed via `pytest`.

## Operational Notes & Tips

* Set the **search language** to match the document language to improve semantic similarity – also for multimodal embeddings.
* The **extraction** process tags each segment’s language and instructs the model to read content in that same language.
* **Rate limits** vary between models; if you use a mixed provider stack, consider applying a rate limiter per model family (see OpenAI limits: [https://platform.openai.com/settings/organization/limits](https://platform.openai.com/settings/organization/limits)).
* **Logging/Debugging:** Exceptions from the extractor are output via the logger.

### Known Exception During PDF Indexing

* When processing large, densely written pages, the model may stop with a `LengthFinishReasonError`.
* **Cause:** Each page is converted into a 200/120-DPI PNG and sent (as base64) together with the prompt and expected structured output. Large image embeddings combined with a long prompt can exceed the model’s response limit.
* **Behavior:** Such pages are logged and skipped during batch runs.


## Model Recommendations for RAG-App
### OpenAI
| Graph     | Step / Component         | Purpose                                          | Recommended Model        | Alternatives                        | When to Use Alternatives                                                                    |
| --------- | ------------------------ | ------------------------------------------------ | ------------------------ | ----------------------------------- | ------------------------------------------------------------------------------------------- |
| Shared    | Embeddings (Chroma)      | Vector representations for retrieval & indexing  | `text-embedding-3-small` | `text-embedding-3-large`            | Use **large** if retrieval quality outweighs cost.                                          |
| Retrieval | `generate_questions`     | Generate query variations in the input language  | `gpt-4o-mini`            | `gpt-4o`, `gpt-4.1`, `gpt-4.1-mini` | Use **full models** for complex/domain-specific queries; `4.1-mini` for a consistent stack. |
| Retrieval | `compress_docs`          | Classify document chunks as relevant or not      | `gpt-4o-mini`            | `gpt-4o`, `gpt-4.1-mini`            | Use **gpt-4o** when relevance is subtle (e.g., legal/medical); `4.1-mini` for lower costs.  |
| Retrieval | `generate_answer`        | Generate final structured answer with references | `gpt-4.1` or `gpt-4o`    | `gpt-4.1-mini`, `gpt-4o-mini`       | Use **mini** if latency/cost is critical and slight quality loss is acceptable.             |
| Index     | `extract` (PDF → chunks) | Multimodal extraction from PDF pages             | `gpt-4o` (multimodal)    | `gpt-4o-mini`                       | Use **mini** for cost-efficient bulk indexing with some OCR noise.                          |
| Index     | `save`                   | Store chunks in Chroma with metadata             | –                        | –                                   | No LLM required.                                                                            |

### Open-Source
| Graph     | Step / Component     | Purpose                                         | Recommended Model              | Alternatives                                                               | When to Use Alternatives                                                                                          |
| --------- | -------------------- | ----------------------------------------------- | ------------------------------ | -------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------- |
| Shared    | Embeddings (Chroma)  | Vector representations for retrieval & indexing | **BAAI/bge-m3** (multilingual) | `intfloat/e5-large-v2`, `jinaai/jina-embeddings-v2`, `nomic-embed-text-v1` | Use **E5** for strong EN quality, **Jina** for long context/multilingual corpora, **Nomic** for top-tier quality. |
| Retrieval | `generate_questions` | Generate query variations                       | **Meta Llama-3.1-8B-Instruct** | `Qwen2.5-7B`, `Mistral-7B`, `Teuken-7B`                                    | Use **Qwen2.5** for tables/code/long context; **Mistral** for speed; **Teuken** for EU/DE support.                |
| Retrieval | `compress_docs`      | Relevance classification per chunk              | **Qwen2.5-7B-Instruct**        | `Llama-3.1-8B`, `Mistral-7B`, smaller `Qwen2.5` variants                   | Use smaller models for throughput or cost; **Llama-3.1-8B** to standardize on Llama.                              |
| Retrieval | `generate_answer`    | Final answer with references                    | **Llama-3.1-70B-Instruct**     | `Llama-3.1-405B` (API), `Qwen2.5-72B`, `Mixtral-8x7B`                      | Use large models for highest quality; **Mixtral/Qwen** for MoE or APAC stacks.                                    |
| Index     | `extract`            | Multimodal extraction                           | **Qwen2.5-VL-7B-Instruct**     | `LLaVA-1.6-34B`, `InternVL2-26B`                                           | Use larger models for complex layouts/charts; **Qwen2.5-VL** for balance of quality & cost.                       |
| Index     | `save`               | Store chunks in Chroma                          | –                              | –                                                                          | No LLM required.                                                                                                  |

### Proprietary Models (Non-OpenAI)
| Graph     | Step / Component     | Purpose                                         | Recommended Model              | Alternatives                                         | When to Use Alternatives                                                                       |
| --------- | -------------------- | ----------------------------------------------- | ------------------------------ | ---------------------------------------------------- | ---------------------------------------------------------------------------------------------- |
| Shared    | Embeddings (Chroma)  | Vector representations for retrieval & indexing | **Voyage AI – voyage-3-large** | `Cohere embed-multilingual-v3.0`, `Gemini Embedding` | If you're already using Cohere/Gemini or need >100 languages.                                  |
| Retrieval | `generate_questions` | Generate semantic query variations              | **Google Gemini 1.5 Flash**    | `Claude Haiku (4.x)`, `Mistral Small (API)`          | Use **Flash** for high query volume/cost focus; others for EU hosting.                         |
| Retrieval | `compress_docs`      | Relevance classification per chunk              | **Gemini 1.5 Flash**           | `Mistral Small`, `Claude Haiku`                      | Use **Flash** for large batch jobs; **Mistral** for EU hosting.                                |
| Retrieval | `generate_answer`    | Generate final, referenced answer               | **Claude 3.5 Sonnet**          | `Gemini Pro`, `Mistral Large 2`, `Cohere Command A`  | Use **Gemini** for long/multimodal context; **Mistral** for multilingual EU needs.             |
| Index     | `extract`            | Multimodal PDF extraction                       | **Gemini 1.5 Pro (Vision)**    | `Gemini Flash`, `Claude 3.5 Vision`, `Voyage MM-3`   | Use **Pro** for complex PDFs; **Flash** for batch jobs; **Voyage** for embedding-first setups. |
| Index     | `save`               | Store chunks in Chroma                          | –                              | –                                                    | No LLM required.                                                                               |

