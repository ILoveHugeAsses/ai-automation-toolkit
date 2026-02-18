# AI Automation Toolkit

Build AI-powered automations with ChatGPT, Claude, and open-source models.

## Features
- **Document Processing:** Summarize, extract, classify PDFs and documents
- **RAG System:** Ask questions about your own data
- **Content Pipeline:** Research -> Draft -> Edit -> Publish
- **Email Automation:** AI-powered email drafting and responses
- **Workflow Engine:** Chain AI tasks with conditional logic

## Quick Start

```python
from ai_toolkit import AIWorkflow

wf = AIWorkflow(model="claude-3")

# Summarize documents
summary = wf.summarize("report.pdf", max_length=500)

# RAG - ask questions about your data
wf.index_documents("./docs/")
answer = wf.ask("What were Q4 revenue numbers?")

# Content pipeline
article = wf.pipeline(
    topic="Web Scraping Best Practices 2025",
    steps=["research", "outline", "draft", "edit"]
)
```

## Components
- `ai_toolkit.summarizer` -- Document summarization
- `ai_toolkit.rag` -- Retrieval-augmented generation  
- `ai_toolkit.pipeline` -- Content generation pipeline
- `ai_toolkit.email` -- Email automation
- `ai_toolkit.workflow` -- Task chaining engine

## License
MIT
