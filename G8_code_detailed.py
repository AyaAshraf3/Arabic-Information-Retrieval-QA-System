# Install dependencies if needed
!pip install --quiet rank_bm25 sentence-transformers faiss-cpu google-generativeai

import numpy as np
import faiss
import re
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
import google.generativeai as genai

# Configure Google Generative AI SDK
genai.configure(api_key='AIzaSyCpS_gcCR3MHkm3Rx0-nCRyIOtQpTKM7xc')
model_gen = genai.GenerativeModel('gemini-2.0-flash-001')

# 1. Load and clean raw text
with open("/kaggle/input/yusuf-story/D8A7D984D983D8AAD8A7D8A8.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

# 2. Clean unwanted tokens
def clean_text(text):
    text = re.sub(r'\bpage_separator\b', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\s*\d+\s*', ' ', text)
    text = re.sub(r'\n+', '\n', text)
    return text.strip()

cleaned_text = clean_text(raw_text)

# 3. Split text into paragraphs
def split_text_into_paragraphs(text, max_sentences=4):
    sentences = re.split(r'(?<=[.؟!\n])\s+', text.strip())
    paragraphs, buffer = [], []
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence: continue
        buffer.append(sentence)
        if len(buffer) >= max_sentences:
            paragraphs.append(' '.join(buffer))
            buffer = []
    if buffer:
        paragraphs.append(' '.join(buffer))
    return paragraphs

paras = split_text_into_paragraphs(cleaned_text)
print(f"📄 Total Paragraphs Extracted: {len(paras)}")

# 4. Save clean paragraphs
with open("/kaggle/working/refined_paragraphs.txt", "w", encoding="utf-8") as f:
    f.write('\n'.join(paras))

# 5. Build BM25 index for display
bm25 = BM25Okapi([p.split() for p in paras])
def search_bm25(q, k=5):
    toks = q.split()
    scores = bm25.get_scores(toks)
    idxs = np.argsort(scores)[::-1][:k]
    return [(float(scores[i]), paras[i]) for i in idxs]

# 6. Build semantic index with Sentence-Transformers
st_model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
retriever     = SentenceTransformer(st_model_name)

print("Embedding paragraphs with Sentence-Transformers…")
st_embs = retriever.encode(paras, convert_to_numpy=True, normalize_embeddings=True)
d = st_embs.shape[1]
faiss.normalize_L2(st_embs)

st_index = faiss.IndexFlatIP(d)
st_index.add(st_embs)
print(f"Semantic index built: {st_index.ntotal} vectors of dim {d}")

def search_semantic(q, k=5):
    qv = retriever.encode([q], convert_to_numpy=True, normalize_embeddings=True)
    D, I = st_index.search(qv, k)
    return [(float(D[0][i]), paras[I[0][i]]) for i in range(k)]

# 7. Define LLM and RAG functions using Gemini
def llm_only_answer(query, max_new_tokens=150):
    prompt = f"السؤال: {query}\nالجواب:"
    response = model_gen.generate_content(
        prompt,
        generation_config=genai.types.GenerationConfig(
            max_output_tokens=max_new_tokens,
            temperature=0.9,
            top_p=0.9
        )
    )
    return response.text.strip()

def rag_answer(query, sem_k=5, max_new_tokens=150):
    sem_hits = search_semantic(query, sem_k)
    contexts = "\n".join(f"- {txt}" for _, txt in sem_hits)
    prompt = (
        "استخدم الفقرات التالية للإجابة على السؤال:\n"
        f"{contexts}\n\n"
        f"السؤال: {query}\nالجواب:"
    )
    response = model_gen.generate_content(
        prompt,
        generation_config=genai.types.GenerationConfig(
            max_output_tokens=max_new_tokens,
            temperature=0.9,
            top_p=0.9
        )
    )
    return response.text.strip()

# 8. Interactive interface
def interactive_rag():
    print("اكتب استعلامك ثم اضغط Enter. اكتب 'exit' للخروج.\n")
    while True:
        q = input("استعلام: ").strip()
        if q.lower() in ("exit","quit",""):
            print("مع السلامة!")
            break

        print("\n>>> نتائج BM25 (للعرض):")
        for score, txt in search_bm25(q, 5):
            print(f"({score:.2f}) {txt}\n")

        print("\n>>> نتائج البحث الدلالي (Sentence-Transformers):")
        for score, txt in search_semantic(q, 5):
            print(f"({score:.3f}) {txt}\n")

        print("\n>>> إجابة بدون استرجاع (LLM only):")
        print(llm_only_answer(q), "\n")

        print(">>> إجابة مع الاسترجاع (RAG semantic only):")
        print(rag_answer(q), "\n")
        print("="*60)

if __name__ == "__main__":
    interactive_rag()