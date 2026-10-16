"""✏️ ส่วนที่ 4: Prompt — บอก LLM ว่าจะให้ทำอะไร

คืนค่าเป็น list ของ message แบบ chat:
    [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}]
(หรือคืนเป็นสตริงก็ได้ ระบบจะถือว่าเป็นข้อความของ user)

hits คือผลจาก retrieval.search เป็น list ของ (chunk, score) อ่านเนื้อหาได้จาก chunk.text และ chunk.doc_id

ค่าตั้งต้นตอนยังไม่ทำ: ไม่เรียก LLM เลย ข้อสอบปรนัยตอบ "ก" ทุกข้อ และโจทย์โค้ดได้ฟังก์ชันว่างที่ raise NotImplementedError
(ทำส่วนนี้ก่อน เพราะถ้ายังไม่ทำ ส่วนอื่นจะไม่มีผลต่อคะแนนเลย)
"""
from student import settings  # noqa: F401 — MAX_CONTEXT_CHARS


def build_mc_prompt(question: str, options: list[tuple[str, str]], hits) -> list[dict]:
    """prompt สำหรับข้อสอบปรนัย

    รับ:  question  คำถาม
          options   list ของ (key, text) เช่น [("ก", "O(1)"), ("ข", "O(log n)"), ...]
          hits      chunk ที่ค้นได้ (list ของ (chunk, score)) อาจเป็น list ว่าง
    คืน:  list ของ message

    หมายเหตุ: core บังคับให้ LLM ตอบเป็น JSON {"reason": "...", "answer": "<key>"} ให้อยู่แล้ว
    แนวทาง:
      - ใส่เนื้อหาของ hits เป็น "เอกสารอ้างอิง" แล้วสั่งให้ยึดเอกสารเป็นหลัก แม้จะขัดกับความรู้ทั่วไป
      - แสดงตัวเลือกพร้อม key (ก. ... ข. ...)
      - ระวังคำถามเชิงปฏิเสธ เช่น "ข้อใด**ไม่**ถูกต้อง"
    """
    # --- สร้าง context จาก hits ---
    context = ""
    if hits:
        char_count = 0
        pieces = []
        for i, (chunk, score) in enumerate(hits, 1):
            piece = f"[{i}] ({chunk.doc_id})\n{chunk.text}"
            if char_count + len(piece) > settings.MAX_CONTEXT_CHARS:
                remaining = settings.MAX_CONTEXT_CHARS - char_count
                if remaining > 100:
                    pieces.append(piece[:remaining])
                break
            pieces.append(piece)
            char_count += len(piece)
        context = "\n---\n".join(pieces)

    # --- จัดรูปแบบตัวเลือก ---
    options_text = "\n".join(f"{key}. {text}" for key, text in options)

    # --- ตรวจจับคำถามเชิงปฏิเสธ ---
    negation_hint = ""
    neg_keywords = ["ไม่", "ยกเว้น", "ผิด", "ไม่ใช่", "ไม่ถูก", "ไม่ได้", "ไม่เป็น"]
    if any(kw in question for kw in neg_keywords):
        negation_hint = (
            "\n⚠️ ข้อควรระวัง: คำถามนี้เป็นคำถามเชิงปฏิเสธ (เช่น ไม่/ยกเว้น/ผิด/ไม่ใช่) "
            "ต้องหาตัวเลือกที่ไม่ตรงกับเอกสาร ผิด หรือเป็นข้อยกเว้น"
        )

    # --- System prompt ---
    system_content = (
        "You are an expert at answering Thai multiple-choice questions based on reference documents.\n"
        "Rules:\n"
        "1. Base your answer STRICTLY on the provided reference documents.\n"
        "2. If the documents conflict with your general knowledge, ALWAYS follow the documents. "
        "For example, if the document explicitly states that lowercase inputs are converted to uppercase and accepted as valid "
        "(e.g. bb-cake-26 is converted to BB-CAKE-26), follow that rule and NEVER claim it is rejected.\n"
        "3. For calculation or reasoning questions, check ALL table columns carefully:\n"
        "   - Check 'ยอดซื้อขั้นต่ำ': if purchase total < minimum purchase, discount is 0.\n"
        "   - Check 'ส่วนลดสูงสุดต่อบิล': if calculated discount exceeds the maximum discount cap, the discount MUST be capped at the maximum discount (e.g. 10% of 1,000 = 100, but cap is 50, so the discount is 50, NOT 100).\n"
        "   - There is NO maximum purchase ceiling unless explicitly stated. Do NOT confuse maximum discount cap with purchase ceiling.\n"
        "4. If the question contains negation (ไม่, ยกเว้น, ผิด, ไม่ใช่), find the option that is incorrect or excluded.\n"
        "5. The choices in 'ตัวเลือก:' are shuffled. Carefully match your chosen answer to the exact key (ก, ข, ค, ง, จ) assigned to it in the 'ตัวเลือก:' list.\n"
        "6. Reply strictly in JSON: {\"reason\": \"detailed step-by-step reasoning in Thai\", \"answer\": \"<exact option key>\"}"
    )

    # --- User prompt ---
    user_parts = []
    if context:
        user_parts.append(f"เอกสารอ้างอิง:\n{context}\n")
    user_parts.append(f"คำถาม: {question}{negation_hint}")
    user_parts.append(f"ตัวเลือก:\n{options_text}")
    if context:
        user_parts.append(
            "คำสั่ง: อ่านเอกสารอ้างอิงอย่างละเอียด:\n"
            "- ตรวจสอบเงื่อนไขยอดซื้อขั้นต่ำและ 'ส่วนลดสูงสุดต่อบิล' ให้ครบถ้วน (หากส่วนลดเกินเพดานส่วนลดสูงสุด ให้ตอบตามเพดานส่วนลด)\n"
            "- ปฏิบัติตามกฎในเอกสารอย่างเคร่งครัด (เช่น การแปลงรหัสตัวพิมพ์เล็กเป็นตัวพิมพ์ใหญ่)\n"
            "- วิเคราะห์ทีละขั้นตอนใน reason ให้ชัดเจน ตรวจสอบว่าตรงกับ key (ก, ข, ค, ง, จ) ใด แล้วระบุ key นั้นใน answer"
        )
    else:
        user_parts.append(
            "เลือกตัวเลือกที่ถูกต้องที่สุด ตรวจสอบ key (ก, ข, ค, ง, จ) ในรายการตัวเลือกให้ตรงกับข้อความที่เลือก"
        )

    user_content = "\n\n".join(user_parts)

    return [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_content},
    ]


def build_code_prompt(requirement: str, signature: str | None, entry_point: str | None, hits) -> list[dict]:
    """prompt สำหรับโจทย์เขียนโปรแกรม

    รับ:  requirement  โจทย์ (ภาษาไทย)
          signature    บรรทัด def ที่ต้องใช้ เช่น "def is_prime(n: int) -> bool:"
          entry_point  ชื่อฟังก์ชัน เช่น "is_prime"
          hits         chunk ที่ค้นได้ (มีเฉพาะโจทย์ที่อ้างถึงเอกสาร ส่วนใหญ่จะเป็น list ว่าง)
    คืน:  list ของ message

    หมายเหตุ: core ดึงโค้ดจาก ```python ... ``` ในคำตอบให้ และถ้า compile ไม่ผ่านจะขอให้ LLM แก้ให้อีก 1 รอบ
    แนวทาง:
      - สั่งให้ใช้ชื่อฟังก์ชันและ signature ตามที่ให้มาเป๊ะ ๆ และใช้เฉพาะ Python standard library
      - ห้าม print หรือรับ input เอง ให้ return ค่า
      - อ่านโจทย์ให้ดี บางข้อ "ห้ามใช้" ฟังก์ชันบางตัว (เช่น max, sorted, eval) ถ้าใช้จะได้ 0 ทั้งข้อ
      - ให้ตอบเป็น code block เดียว
    """
    # --- System prompt ---
    system_content = (
        "You are an expert Python software engineer. Write correct, robust, and clean Python code.\n"
        "Rules:\n"
        "1. Use ONLY Python standard library (e.g. re, math, etc.). No external libraries.\n"
        "2. Do NOT use print() or input(). The function MUST return the result.\n"
        "3. Follow the exact function signature and entry_point provided.\n"
        "4. If the requirement forbids certain functions or attributes (e.g. max, sorted, eval), do NOT use them.\n"
        "5. ALWAYS import all needed standard modules at the top of the code (e.g. `import re`).\n"
        "6. ALWAYS initialize return/calculation variables (e.g. `discount = 0.0`) at the very top of the function.\n"
        "7. Adhere strictly to the reference documents and business logic:\n"
        "   - Normalize string codes first: `code = str(code).strip().upper()`.\n"
        "   - Minimum purchase condition: 'ยอดซื้อต้องไม่น้อยกว่ายอดซื้อขั้นต่ำ' means if `total < min_spend`: return 0.0 (or discount remains 0.0). Note that `total >= min_spend` is valid!\n"
        "   - Maximum discount cap ('ส่วนลดสูงสุดต่อบิล'): If a numerical cap is specified, the discount cannot exceed it: `if cap is not None and discount > cap: discount = float(cap)`. If cap is 'ไม่มี' or None, do NOT limit.\n"
        "   - NEVER compare a number with None (e.g. avoid `total >= None` or `discount > None`).\n"
        "   - Follow negative instructions (e.g. if requirement says 'ไม่ต้องตรวจสถานะสมาชิกและวันหมดอายุ', do NOT check member level or expiry).\n"
        "   - Return types: Ensure float for monetary amounts/discounts (e.g. `return float(round(discount, 2))`), bool for validator, int for count.\n"
        "8. Reply with a SINGLE ```python code block containing the imports and the function definition. No explanations."
    )

    # --- User prompt ---
    user_parts = []

    # เพิ่ม context จาก hits (ถ้ามี)
    if hits:
        ctx_pieces = []
        char_count = 0
        for chunk, score in hits:
            piece = chunk.text
            if char_count + len(piece) > settings.MAX_CONTEXT_CHARS:
                break
            ctx_pieces.append(piece)
            char_count += len(piece)
        if ctx_pieces:
            ref_docs = "\n\n".join(ctx_pieces)
            user_parts.append(f"Reference Documents:\n{ref_docs}")

    user_parts.append(f"Requirement:\n{requirement}")

    if signature:
        user_parts.append(f"Function signature (use exactly):\n{signature}")
    elif entry_point:
        user_parts.append(f"Function name: {entry_point}")

    user_parts.append(
        "Write complete, working Python code.\n"
        "- Include `import re` at the top if regex is used.\n"
        "- Normalize input: `code = str(code).strip().upper()`.\n"
        "- Check minimum purchase: `if total < min_spend: return 0.0`.\n"
        "- Apply maximum discount limit per bill (ส่วนลดสูงสุดต่อบิล) if specified (handle None / ไม่มี safely).\n"
        "- Return float for discount.\n"
        "Respond with a single ```python code block containing the imports and function definition."
    )

    user_content = "\n\n".join(user_parts)

    return [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_content},
    ]
