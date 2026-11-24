from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

REFERENCE = Path(__file__).resolve().parent / "data" / "lingqian.md"
EXPECTED_FIELDS = (
    "象意",
    "签名与等级",
    "卦辞",
    "推断",
    "大象",
    "运势",
    "爱情",
    "疾病",
    "失物",
    "诉讼",
)
HEADING = re.compile(r"^## 第(\d+)卦 (.+?)｜编码 ([01]{6})[ \t]*$", re.MULTILINE)
FIELD = re.compile(r"^- ([^：\n]+)：", re.MULTILINE)


@dataclass(frozen=True)
class Entry:
    number: int
    name: str
    code: str
    body: str


def load_entries(path: Path) -> dict[str, Entry]:
    text = path.read_text(encoding="utf-8-sig")
    matches = list(HEADING.finditer(text))
    entries = {}
    for index, match in enumerate(matches):
        number, name, code = match.groups()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        body = re.sub(r"\n+---\s*$", "", text[match.end() : end]).strip("\n")
        fields = FIELD.findall(body)
        missing = [field for field in EXPECTED_FIELDS if field not in fields]
        if missing:
            raise ValueError(
                f"《灵签详解》{name}（{code}）缺少字段：{'、'.join(missing)}"
            )
        if code in entries:
            raise ValueError(f"《灵签详解》存在重复卦码：{code}")
        entries[code] = Entry(int(number), name, code, body)
    if len(entries) != 64:
        raise ValueError(f"《灵签详解》应有 64 条，实际读取 {len(entries)} 条。")
    return entries


@lru_cache(maxsize=1)
def default_entries() -> dict[str, Entry]:
    return load_entries(REFERENCE)


def format_references(layers: dict[str, str], entries: dict[str, Entry]) -> str:
    by_code: dict[str, list[str]] = {}
    for layer, code in layers.items():
        by_code.setdefault(code, []).append(layer)

    blocks = []
    for code, names in by_code.items():
        layers_label = "对应层次：" + "、".join(names)
        entry = entries.get(code)
        if entry is None:
            blocks.append(f"{layers_label}\n知识库未检索到对应条目（{code}）。")
            continue
        title = f"《灵签详解》｜第{entry.number}卦 {entry.name}｜编码{entry.code}"
        blocks.append(f"{title}\n{layers_label}\n{entry.body}")
    return "\n\n".join(blocks)
