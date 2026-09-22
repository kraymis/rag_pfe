# RAG PFE
## PFE RAG Assistant

Install the dependencies and start the Streamlit interface from the project root:

```bash
pip install -r requirements.txt
streamlit run src/app.py
```

Set `NVIDIA_API_KEY` in `.env` before starting the application. The FAISS index
and `chunks.json` must already be present in `index/`.
