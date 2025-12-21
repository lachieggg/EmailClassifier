# Default host for Ollama API
DEFAULT_OLLAMA_HOST = 'http://localhost'

# Default port for Ollama API
DEFAULT_OLLAMA_PORT = 11434

# Default prompt template for email classification
DEFAULT_PROMPT = """You are an email classifier helping filter unwanted or spam emails.

Classify this email into ONE category:

1. DELETE
2. KEEP

Email:
---
{email_text}
---

Respond in this EXACT format:
CATEGORY: [UNWANTED/NORMAL]
CONFIDENCE: [HIGH/MEDIUM/LOW]
REASON: [Brief explanation in one sentence]"""