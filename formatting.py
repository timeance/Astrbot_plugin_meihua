"""Convert common model Markdown to readable text for plain chat messages."""

import re


def plain_text(text: str) -> str:
    protected: list[str] = []

    def keep(value: str) -> str:
        protected.append(value)
        return f"\x00{len(protected) - 1}\x00"

    # Code contents and escaped punctuation are literal, not emphasis or headings.
    text = re.sub(
        r"^ {0,3}(?P<fence>`{3,}|~{3,})[^\n]*\n(?P<body>.*?)"
        r"(?:^ {0,3}(?P=fence)[ \t]*(?=\n|$)|\Z)",
        lambda match: keep(match["body"].rstrip("\n")),
        text,
        flags=re.M | re.S,
    )
    text = re.sub(
        r"(?P<ticks>`+)(?P<body>[^\n]+?)(?P=ticks)(?!`)",
        lambda match: keep(match["body"]),
        text,
    )
    text = re.sub(r"\\([\\`*_{}\[\]()#+.!>|~-])", lambda match: keep(match[1]), text)
    text = re.sub(
        r"!?\[([^\]\n]+)\]\((https?://[^\s()]+(?:\([^\s()]*\)[^\s()]*)*)\)",
        lambda match: f"{match[1]}（{keep(match[2])}）",
        text,
    )
    lines = []
    for line in text.splitlines():
        line = re.sub(r"^ {0,3}(?:>[ \t]?)+", "", line)
        if re.fullmatch(
            r"\s*(?:(?:\*[ \t]*){3,}|(?:-[ \t]*){3,}|(?:_[ \t]*){3,}|={3,})\s*", line
        ):
            continue
        if re.match(r"^ {0,3}#{1,6}(?:\s|$)", line):
            line = re.sub(r"^ {0,3}#{1,6}[ \t]*", "", line)
            line = re.sub(r"[ \t]+#+[ \t]*$", "", line)
        line = re.sub(r"^(\s*)[-+*][ \t]+", r"\1• ", line)
        lines.append(line)
    text = "\n".join(lines)
    for marker in (r"\*\*\*", "___", r"\*\*", "__", "~~"):
        text = re.sub(
            rf"(?<!\w){marker}(.+?){marker}(?!\w)"
            if "_" in marker
            else rf"{marker}(.+?){marker}",
            r"\1",
            text,
            flags=re.S,
        )
    text = re.sub(r"(?<!\*)\*(?![\s*])([^*\n]+?)\*(?!\*)", r"\1", text)
    text = re.sub(r"(?<!\w)_(?![\s_])([^_\n]+?)_(?!\w)", r"\1", text)
    text = re.sub(r"\n[ \t]*\n(?:[ \t]*\n)+", "\n\n", text).strip()
    return re.sub(r"\x00(\d+)\x00", lambda match: protected[int(match[1])], text)
