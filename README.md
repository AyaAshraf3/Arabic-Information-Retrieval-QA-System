# Arabic-Information-Retrieval-QA-System
Developed a full Arabic QA pipeline using Retrieval-Augmented Generation (RAG) with classical and semantic search. Built around a public Arabic text ("قصة يوسف عليه السلام"), the system answers Arabic queries using LLMs grounded in retrieved passages.
Built a full-stack Arabic QA pipeline using semantic retrieval and Retrieval-Augmented Generation (RAG).

Selected and preprocessed an Arabic public-domain book ("قصة يوسف عليه السلام في القرآن") by cleaning and chunking text into semantically coherent 2–4 sentence paragraphs.

Engineered a multilingual embedding pipeline using SentenceTransformers (paraphrase-multilingual-MiniLM-L12-v2) for dense vector generation across Arabic texts.

Indexed over 700 vectorized chunks using FAISS (IndexFlatIP) for efficient semantic retrieval based on cosine similarity.

Designed a hybrid retrieval system supporting both classical keyword-based search (BM25) and semantic search; evaluated top-5 hits for 10 Arabic natural-language queries.

Integrated Google Gemini via API to perform RAG (retrieval + generation) and LLM-only answer generation for open-ended Arabic questions.

Benchmarked and compared answer relevance between classical search, semantic search, RAG-enhanced responses, and vanilla LLM responses.

Demonstrated that RAG improved grounding, accuracy, and contextual detail over LLM-only generations, especially for Quranic and literary Arabic queries.

Delivered full-code solution with reproducible outputs and modular inference pipeline ready for extension or deployment.
