# Trade Compliance Assistant — Basic

This simple project contains:

- Chroma persistent vector DB
- OpenAI embeddings
- Pydantic metadata
- Top-K retrieval
- CrossEncoder reranking
- exactly two LLM-callable tools
- LangGraph tool loop
- try/except error handling

## Architecture

```text
User
 |
 v
LLM
 |
 +-------------------------------+
 |                               |
 v                               v
Tool 1                          Tool 2
search_trade_documents         check_trade_compliance
 |                               |
 v                               v
Chroma Top-K                  Chroma metadata lookup
 |
 v
CrossEncoder reranking
 |
 v
Top-N
 |
 +---------------+
                 |
                 v
                LLM
                 |
                 v
            Final answer
```

## Only two tools

```python
TOOLS = [
    search_trade_documents,
    check_trade_compliance,
]

llm_with_tools = llm.bind_tools(TOOLS)
```

The LLM decides which tool to call.

There is no hard-coded routing like:

```python
if "invoice" in question:
    call_tool()
```

## Top-K and reranking

Example:

```env
TOP_K=5
RERANK_TOP_N=3
```

Flow:

```text
User query
   |
   v
Chroma
   |
   | retrieve Top 5
   v
CrossEncoder
   |
   | rerank
   v
Best 3 documents
   |
   v
LLM
```

## Project structure

```text
trade-compliance-basic-v2/
|
+-- app/
|   +-- __init__.py
|   +-- config.py
|   +-- models.py
|   +-- vector_store.py
|   +-- reranker.py
|   +-- tools.py
|   +-- graph.py
|   +-- main.py
|
+-- .env.example
+-- requirements.txt
+-- README.md
```

## Run

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

Install:

```bash
pip install -r requirements.txt
```

Copy `.env.example` to `.env`.

Then:

```bash
python -m app.main
```

## Example questions

```text
What is the Incoterm for SHP1001?
```

LLM should use:

```text
search_trade_documents
```

For:

```text
Is SHP1002 compliant and why?
```

LLM should use:

```text
check_trade_compliance
```

For:

```text
What is an Incoterm?
```

The LLM can answer directly without a tool.

## Interview explanation

> I store shipment and invoice data in Chroma with metadata.
> At query time Chroma retrieves Top-K candidate documents.
> Then I use a CrossEncoder to rerank those candidates and keep
> Top-N before sending the retrieved information to the LLM.
> I bind exactly two tools to the LLM: one retrieval tool and
> one deterministic compliance-check tool. LangGraph ToolNode
> executes whichever tool the LLM selects. I use try/except at
> the vector DB, reranker, tool, LLM, graph, and application levels
> so failures are handled safely.
