# Demo_Self-ERP_RAG
RAG demo implementation based on LlamaIndex
for chat with Self-ERP data documentation.

1. Documents should be placed in the "data" directory (one sample document included)
2. OpenAI api key must be added to .env file.
3. Prerequisite: Python
4. Instalation: pip install -r requirements.txt
5. Execution: streamlit run demo.py
6. During the first execution files in the "data" directory will be converted into a vector index that will be saved in new directory called "cache".
