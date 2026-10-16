"""✏️ ส่วนที่ 1: Chunking — แบ่งเอกสารเป็นชิ้น (chunk)

เอกสารที่ได้รับเป็น markdown เช่น lecture note หรือ case study ที่มีหัวข้อ (#, ##), ตาราง และ code block
ชิ้นที่แบ่งแล้วจะถูกนำไปทำ embedding และใช้ค้นหาตอนตอบคำถาม
แบ่งดี = ค้นเจอเนื้อหาที่ตรงคำถามและครบถ้วน

ค่าตั้งต้นตอนยังไม่ทำ: ไม่แบ่งเลย ทั้งเอกสารเป็น chunk เดียว
ค่าที่ปรับได้ (CHUNK_SIZE, CHUNK_OVERLAP, ...) อยู่ใน student/settings.py
"""
import re

from student import settings  # noqa: F401


def split(doc_id: str, text: str) -> list[str]:
    """แบ่งเอกสาร 1 ชิ้นเป็น list ของข้อความ

    รับ:  doc_id  ชื่อเอกสาร เช่น "session04_01_prompt_fundamentals"
          text    เนื้อหาทั้งหมดของเอกสาร (markdown)
    คืน:  list[str] ข้อความของแต่ละ chunk (ห้ามว่าง)

    แนวทาง:
      - แยกตามหัวข้อ markdown (บรรทัดที่ขึ้นต้นด้วย #) แล้วใส่ชื่อหัวข้อไว้ต้น chunk ด้วย
      - หัวข้อที่ยาวเกินไปให้ตัดย่อยเป็นช่วง ๆ (เช่น 600–1,000 ตัวอักษร) และซ้อนกันเล็กน้อย (overlap)
      - พยายามอย่าตัดกลางตาราง กลาง code block หรือกลางประโยค
    ตัวอย่าง:
      split("doc1", "# A\\nเนื้อหา A\\n# B\\nเนื้อหา B") -> ["A\\nเนื้อหา A", "B\\nเนื้อหา B"]
    """
    if not text or not text.strip():
        return [text or ""]

    if settings.SPLIT_BY_HEADING:
        sections = _split_by_heading(text)
    else:
        sections = [("", text)]

    chunks: list[str] = []
    for heading, content in sections:
        if settings.INCLUDE_HEADING and heading:
            full_text = f"{heading}\n{content}".strip()
        else:
            full_text = content.strip()

        if not full_text:
            continue

        if len(full_text) <= settings.CHUNK_SIZE:
            chunks.append(full_text)
        else:
            sub_chunks = _split_long_text(
                full_text, settings.CHUNK_SIZE, settings.CHUNK_OVERLAP
            )
            chunks.extend(sub_chunks)

    # ห้ามคืน list ว่าง
    return chunks if chunks else [text]


def _split_by_heading(text: str) -> list[tuple[str, str]]:
    """แยก markdown ตามหัวข้อ (# ## ###...) คืน list ของ (breadcrumb_heading, body_text)."""
    lines = text.split("\n")
    sections: list[tuple[str, str]] = []
    current_heading = ""
    current_lines: list[str] = []
    heading_stack: list[tuple[int, str]] = []  # (level, title_text)

    for line in lines:
        m = re.match(r"^(#{1,6})\s+(.*)$", line)
        if m:
            level = len(m.group(1))
            heading_title = m.group(2).strip()

            # เจอหัวข้อใหม่ → เก็บ section ก่อนหน้า
            if current_lines or current_heading:
                sections.append((current_heading, "\n".join(current_lines)))

            # อัปเดต stack: pop หัวข้อที่มี level >= level ปัจจุบัน
            while heading_stack and heading_stack[-1][0] >= level:
                heading_stack.pop()
            heading_stack.append((level, heading_title))

            # สร้าง breadcrumb เช่น "[คู่มือระบบคูปอง > 2. ประเภทคูปอง]"
            breadcrumb = " > ".join(t for _, t in heading_stack)
            current_heading = f"[{breadcrumb}]\n{line.strip()}"
            current_lines = []
        else:
            current_lines.append(line)

    # section สุดท้าย
    if current_lines or current_heading:
        sections.append((current_heading, "\n".join(current_lines)))

    return sections if sections else [("", text)]


def _split_long_text(text: str, max_size: int, overlap: int) -> list[str]:
    """ตัดข้อความยาวเป็น chunk ย่อย พยายามตัดที่ย่อหน้าหรือบรรทัดว่าง
    ไม่ตัดกลาง code block หรือตาราง."""
    # แยกเป็นบล็อกย่อย (ย่อหน้า, code block, ตาราง)
    blocks = _extract_blocks(text)

    chunks: list[str] = []
    current_chunk = ""

    for block in blocks:
        if not block.strip():
            continue

        # ถ้าเพิ่มบล็อกนี้แล้วเกินขนาด → บันทึก chunk เดิมก่อน
        if current_chunk and len(current_chunk) + len(block) + 2 > max_size:
            chunks.append(current_chunk.strip())
            # เก็บ overlap จากท้าย chunk เดิม
            if overlap > 0 and len(current_chunk) > overlap:
                tail = current_chunk[-overlap:]
                current_chunk = tail + "\n\n" + block
            else:
                current_chunk = block
        else:
            current_chunk = (current_chunk + "\n\n" + block) if current_chunk else block

    if current_chunk.strip():
        chunks.append(current_chunk.strip())

    return chunks if chunks else [text]


def _extract_blocks(text: str) -> list[str]:
    """แยกข้อความเป็นบล็อก โดยรักษา code block (```) และตาราง (|) ไว้เป็นชิ้นเดียว."""
    blocks: list[str] = []
    current: list[str] = []
    in_code_block = False
    in_table = False

    for line in text.split("\n"):
        # ตรวจ code block
        if line.strip().startswith("```"):
            if in_code_block:
                # ปิด code block
                current.append(line)
                blocks.append("\n".join(current))
                current = []
                in_code_block = False
                continue
            else:
                # เปิด code block → เก็บข้อมูลก่อนหน้า
                if current and not in_table:
                    blocks.append("\n".join(current))
                    current = []
                current.append(line)
                in_code_block = True
                continue

        if in_code_block:
            current.append(line)
            continue

        # ตรวจตาราง (บรรทัดที่ขึ้นต้นด้วย |)
        is_table_line = line.strip().startswith("|")
        if is_table_line:
            if not in_table and current:
                blocks.append("\n".join(current))
                current = []
            in_table = True
            current.append(line)
            continue
        elif in_table:
            # จบตาราง
            blocks.append("\n".join(current))
            current = []
            in_table = False

        # บรรทัดว่าง = จุดแบ่งย่อหน้า
        if not line.strip():
            if current:
                blocks.append("\n".join(current))
                current = []
        else:
            current.append(line)

    if current:
        blocks.append("\n".join(current))

    return blocks
