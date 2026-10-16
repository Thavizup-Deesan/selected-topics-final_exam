"""✏️ ส่วนที่ 3: Retrieval — ค้นหา chunk ที่เกี่ยวข้องกับคำถาม

ค่าตั้งต้นตอนยังไม่ทำ: ไม่คืน chunk ใดเลย LLM จึงไม่เห็นเอกสาร
"""
import re

from core.types import Chunk
from core.utils import cosine_many  # noqa: F401 — ตัวช่วยคำนวณความคล้ายของ vector
from student import settings  # noqa: F401 — TOP_K, QUERY_WITH_OPTIONS, HYBRID_WEIGHT, MIN_SCORE


def make_query(question: str, options: list[str]) -> str:
    """สร้างข้อความที่จะใช้ค้นหา

    รับ:  question  คำถาม (หรือ requirement ของโจทย์เขียนโค้ด ซึ่งกรณีนี้ options จะเป็น list ว่าง)
          options   ข้อความของตัวเลือก ก–จ (ไม่มีตัวอักษร key นำหน้า)
    คืน:  สตริงคำค้น (ห้ามว่าง)

    แนวทาง: ลองรวมคำถามกับตัวเลือกเข้าด้วยกัน เพราะคำตอบมักอยู่ในย่อหน้าที่มีคำของตัวเลือก
    """
    if settings.QUERY_WITH_OPTIONS and options:
        return question + " " + " ".join(options)
    return question


def search(
    chunks: list[Chunk],
    query_text: str,
    query_vector: list[float] | None,
    k: int,
) -> list[tuple[Chunk, float]]:
    """เลือก chunk ที่เกี่ยวข้องที่สุด k ชิ้น

    รับ:  chunks        chunk ทั้งหมดของเอกสารชุดนี้ (chunk.text, chunk.vector, chunk.doc_id)
          query_text    คำค้น (จาก make_query)
          query_vector  vector ของคำค้น (จาก embedding.embed_query) หรือ None ถ้ายังไม่ได้ทำ embedding
          k             จำนวนที่ระบบแนะนำ (5) จะใช้ settings.TOP_K ของตัวเองแทนก็ได้
    คืน:  list ของ (chunk, score) เรียงจากเกี่ยวข้องมากไปน้อย

    แนวทาง:
      - วิธีพื้นฐาน: scores = cosine_many(query_vector, [c.vector for c in chunks]) แล้วเลือก k ตัวที่คะแนนสูงสุด
      - วิธีที่ดีขึ้น: hybrid search คือรวมคะแนน vector กับคะแนนการจับคำ (keyword / BM25) ช่วยมากกับชื่อเฉพาะและตัวเลข
      - ระวังกรณีที่ chunk.vector หรือ query_vector เป็น None
    """
    if not chunks:
        return []

    top_k = settings.TOP_K or k

    # --- Vector scores (cosine similarity) ---
    vector_scores = [0.0] * len(chunks)
    if query_vector is not None:
        vectors = [c.vector for c in chunks]
        if all(v is not None for v in vectors):
            vector_scores = cosine_many(query_vector, vectors)

    # --- Keyword / BM25 scores (สำหรับ hybrid search) ---
    keyword_scores = [0.0] * len(chunks)
    hybrid_w = settings.HYBRID_WEIGHT
    if hybrid_w > 0:
        keyword_scores = _bm25_scores(chunks, query_text)

    # --- รวมคะแนน ---
    final_scores = [
        (1 - hybrid_w) * vs + hybrid_w * ks
        for vs, ks in zip(vector_scores, keyword_scores)
    ]

    # --- เรียงและเลือก top-k ---
    scored = sorted(
        zip(chunks, final_scores), key=lambda x: x[1], reverse=True
    )

    # --- กรองคะแนนต่ำ ---
    if settings.MIN_SCORE is not None:
        scored = [(c, s) for c, s in scored if s >= settings.MIN_SCORE]

    return scored[:top_k]


# =====================================================================
# Helper functions
# =====================================================================

def _tokenize(text: str) -> list[str]:
    """Tokenize สำหรับ BM25 — รองรับคำภาษาอังกฤษ ตัวเลข และ character n-grams สำหรับภาษาไทย."""
    tokens = []
    # 1. คำภาษาอังกฤษและตัวเลข
    eng_num = re.findall(r"[a-z0-9_]+", text.lower())
    tokens.extend(eng_num)

    # 2. ภาษาไทย: ดึงช่วงตัวอักษรภาษาไทยแล้วสร้าง 2-gram และ 3-gram เพื่อให้ BM25 จับคำไทยได้แม่นยำ
    thai_parts = re.findall(r"[\u0E00-\u0E7F]+", text)
    for part in thai_parts:
        if len(part) <= 2:
            tokens.append(part)
        else:
            for i in range(len(part) - 1):
                tokens.append(part[i:i + 2])
            for i in range(len(part) - 2):
                tokens.append(part[i:i + 3])
    return tokens


def _bm25_scores(chunks: list[Chunk], query_text: str) -> list[float]:
    """คำนวณ BM25 scores แล้ว normalize เป็น [0, 1]."""
    try:
        from rank_bm25 import BM25Okapi

        tokenized_corpus = [_tokenize(c.text) for c in chunks]
        tokenized_query = _tokenize(query_text)
        if not tokenized_query:
            return [0.0] * len(chunks)
        bm25 = BM25Okapi(tokenized_corpus)
        raw = bm25.get_scores(tokenized_query)
        max_s = max(raw) if len(raw) > 0 and max(raw) > 0 else 1.0
        return [float(s / max_s) for s in raw]
    except Exception:
        # Fallback: simple keyword overlap
        return _keyword_overlap_scores(chunks, query_text)


def _keyword_overlap_scores(chunks: list[Chunk], query_text: str) -> list[float]:
    """Fallback: นับคำ/n-gram ที่ตรงกันแล้ว normalize."""
    query_tokens = set(_tokenize(query_text))
    if not query_tokens:
        return [0.0] * len(chunks)
    scores = []
    for c in chunks:
        chunk_tokens = set(_tokenize(c.text))
        overlap = len(query_tokens & chunk_tokens)
        scores.append(overlap / len(query_tokens))
    max_s = max(scores) if scores and max(scores) > 0 else 1.0
    return [s / max_s for s in scores]
