from __future__ import annotations

import json
import re
from dataclasses import dataclass
from html import unescape

from .casting import (
    BRANCHES,
    BY_CODE,
    LINE_NAMES,
    STEMS,
    TRIGRAMS,
    SiteTime,
    cast_numbers,
    changed_code,
    moving_between,
    mutual_code,
    site_relation,
    trigram_details,
)
from .knowledge import Entry, default_entries, format_references


class InputError(ValueError):
    """A user-facing correction for the current command."""


@dataclass(frozen=True)
class PreparedReading:
    summary: str
    references: str
    model_input: str


LAYERS = ("本卦", "互卦", "变卦")
MISSING = "请补充报数，或提供本卦和变卦／动爻，然后重发完整命令。"
FIELD_NAMES = {
    "占问": "占问",
    "事由": "占问",
    "问题": "占问",
    "时间": "时间",
    "四柱": "四柱",
    "空亡": "空亡",
    "旬空": "空亡",
    "报数": "报数",
    "起卦数字": "报数",
    "本卦": "本卦",
    "卦码": "本卦",
    "编码": "本卦",
    "互卦": "互卦",
    "变卦": "变卦",
    "动爻": "动爻",
    "上卦": "上卦",
    "下卦": "下卦",
    "体用": "体用",
    "起卦法": "方法",
    "起卦方式": "方法",
    "方法": "方法",
    "来源": "来源",
    "节气": "节气",
    "设置": "设置",
    "旺衰基准": "旺衰基准",
    "库基准": "库基准",
    "空亡基准": "空亡基准",
}
FIELD = re.compile(r"^(" + "|".join(FIELD_NAMES) + r")(?:\s*[:：]\s*|\s+|$)(.*)$")
FENCE = re.compile(r"^(?:`{3,}|~{3,})[\w+-]*\s*$")
EXPLICIT_TRIGRAM = re.compile(r"^(上卦|下卦)\s*取\s*(.+)$")
SITE_SOURCE = re.compile(
    r"https?://meihua\.iztro\.com(?:/\S*)?$|^meihua\.iztro\.com/?$", re.I
)
METHOD_LINE = re.compile(
    r"(?:采用|使用|用)?\s*(时间起卦|铜钱法|蓍草法|手动起卦|手动排盘|报数起卦)"
)
NUMBER_LIST = r"[0-9]+(?:[\s,，、]+[0-9]+)*"
BARE_NUMBERS = re.compile(rf"^({NUMBER_LIST})(?:[ \t]+(.+))?$")
NUMERIC_CONTEXT = re.compile(
    r"^(?:[+\-−￥¥$0-9%％]|[年月日时点分秒岁元块万亿成折天周个笔条套期](?!间|钟|子|旦|末|刊|记|历|志))"
)
LINE_VALUE = re.compile(r"(?:第)?(初|[一二三四五六上]|[1-6])(?:爻(?:动)?)?")
LINE_NUMBER = dict(zip("初一二三四五六上", (1, 1, 2, 3, 4, 5, 6, 6)))
PILLAR = re.compile(f"[{STEMS}][{BRANCHES}]")
VOID = re.compile(f"[{BRANCHES}]{{2}}")
SEPARATORS = re.compile(r"[\s,，、/／]+")


def _moving_lines(value: str) -> tuple[int, ...]:
    if value in {"无", "无动爻", "静卦", "不动"}:
        return ()
    parts = SEPARATORS.split(value.strip()) if value.strip() else []
    matches = [LINE_VALUE.fullmatch(part) for part in parts]
    if not matches or any(match is None for match in matches):
        raise InputError("动爻请填写初爻至上爻（如「初爻、五爻」），然后重发。")
    parts = [match[1] for match in matches if match is not None]
    lines = [LINE_NUMBER[part] if part in LINE_NUMBER else int(part) for part in parts]
    if len(set(lines)) != len(lines):
        raise InputError("动爻有重复，请核对后重发。")
    return tuple(sorted(lines))


def _parse(text: str) -> tuple[dict[str, str], str]:
    fields: dict[str, str] = {}
    free: list[str] = []
    lines = [
        line.strip().removesuffix("\\").strip() for line in unescape(text).splitlines()
    ]
    lines = [line for line in lines if line and not FENCE.fullmatch(line)]

    def put(key: str, value: str) -> None:
        if key in {"四柱", "空亡", "报数"}:
            value = " ".join(value.split())
        if key in fields and fields[key] != value:
            raise InputError(f"「{key}」重复且不一致，请核对后重发。")
        fields[key] = value

    def read_row(index: int, pattern: re.Pattern, value: str = "") -> tuple[str, int]:
        cells = value.split()
        while len(cells) < 4 and all(pattern.fullmatch(cell) for cell in cells):
            if index >= len(lines):
                break
            more = lines[index].split()
            if len(cells) + len(more) > 4 or not all(
                pattern.fullmatch(cell) for cell in more
            ):
                break
            cells.extend(more)
            index += 1
        return " ".join(cells), index

    index = 0
    while index < len(lines):
        line = lines[index]
        index += 1
        if EXPLICIT_TRIGRAM.fullmatch(line):
            for part in re.split(r"[、,，；;]", line):
                part = part.strip()
                trigram = EXPLICIT_TRIGRAM.fullmatch(part)
                if trigram:
                    put(trigram[1], trigram[2])
                elif part.endswith("爻动") and LINE_VALUE.fullmatch(part):
                    put("动爻", part)
            continue
        field = FIELD.fullmatch(line)
        if field:
            key, value = field.groups()
            key = FIELD_NAMES[key]
            # Clipboard tables repeat "时间" as the header for the four pillars.
            if key == "时间":
                header, next_index = read_row(index, re.compile("[年月日时]"), value)
                if header == "年 月 日 时":
                    pillars, index = read_row(next_index, PILLAR)
                    if len(pillars.split()) == 4:
                        put("四柱", pillars)
                    continue
            if key in {"四柱", "空亡"}:
                value, index = read_row(index, PILLAR if key == "四柱" else VOID, value)
            elif not value and index < len(lines) and not FIELD.fullmatch(lines[index]):
                value = lines[index]
                index += 1
            if not value:
                if key in {*LAYERS, "动爻", "报数", "上卦", "下卦"}:
                    raise InputError(f"「{key}」缺少内容，请补齐后重发完整命令。")
                continue
            put(key, value)
            continue
        if SITE_SOURCE.fullmatch(line):
            put("来源", line)
        elif method := METHOD_LINE.fullmatch(line):
            put("方法", method[1])
        elif not fields and not free:
            free.append(line)
    return fields, free[0] if free else ""


def _number_input(value: str, bare: str) -> tuple[tuple[int, ...], str]:
    if value:
        if not re.fullmatch(NUMBER_LIST, value):
            raise InputError("报数须为正整数（例如 11 12），请修正后重发。")
        numbers, question = value, ""
    else:
        match = BARE_NUMBERS.fullmatch(bare)
        if not match:
            return (), ""
        numbers, question = match[1], match[2] or ""
        if re.fullmatch(r"[01]{6}", numbers):
            raise InputError(
                "这是六位卦码，还是一个报数？请标明「卦码」或「报数」后重发。"
            )
        if NUMERIC_CONTEXT.match(question):
            return (), ""
    try:
        values = tuple(int(item) for item in re.findall(r"[0-9]+", numbers))
    except ValueError as exc:
        raise InputError("报数过大，请改用较小的正整数后重发。") from exc
    question_field = FIELD.fullmatch(question)
    if question_field and FIELD_NAMES[question_field[1]] == "占问":
        question = question_field[2]
    return values, question


def _hexagram(value: str, entries: dict[str, Entry]) -> str:
    value = value.strip(" `")
    if value in entries:
        return value
    explicit = re.fullmatch(r"(.+?)[\s(（]+(?:编码\s*)?([01]{6})[)）]?", value)
    name = explicit[1].strip() if explicit else value
    exact = [entry.code for entry in entries.values() if entry.name == name]
    aliases = [
        entry.code
        for entry in entries.values()
        if entry.name[2:] == name
        or (entry.code[:3] == entry.code[3:] and entry.name[0] == name)
    ]
    choices = exact or aliases
    if len(choices) != 1:
        raise InputError("卦名无法唯一识别，请填写完整卦名或六位卦码后重发。")
    if explicit and explicit[2] != choices[0]:
        raise InputError("卦名与六位卦码不一致，请核对后重发。")
    return choices[0]


def _trigram_code(value: str) -> str:
    for index, gua in enumerate(TRIGRAMS, 1):
        if value in {str(index), gua.name, gua.code}:
            return gua.code
    raise InputError("请填写有效的上卦、下卦（卦名或 1 至 8）后重发。")


def _time_context(fields: dict[str, str]) -> tuple[SiteTime | None, str]:
    unverified = "本次未纳入时间标签"
    pillars = re.sub(r"[\s,，、]+", "", fields.get("四柱", ""))
    empty = re.sub(r"[\s,，、]+", "", fields.get("空亡", ""))
    if not re.fullmatch(f"(?:{PILLAR.pattern}){{4}}", pillars):
        return None, unverified
    if not re.fullmatch(f"(?:{VOID.pattern}){{4}}", empty):
        return None, unverified
    selectors = ("旺衰基准", "库基准", "空亡基准")
    picked = tuple(
        fields.get(key, default) for key, default in zip(selectors, "月月日")
    )
    picked = tuple(re.sub(r"(?:支|柱(?:旬空)?)$", "", value) for value in picked)
    if any(value not in tuple("年月日时") for value in picked) or (
        fields.get("设置") and not all(fields.get(key) for key in selectors)
    ):
        return None, unverified + "（参照设置待确认）"
    strength, storage, void = ("年月日时".index(value) * 2 for value in picked)
    time = SiteTime(
        pillars[strength + 1],
        pillars[storage + 1],
        pillars[4:6],
        empty[void : void + 2],
    )
    prefix = "网站设置" if any(key in fields for key in selectors) else "网站默认"
    strength_name, storage_name, void_name = picked
    basis = f"{prefix}：旺衰看{strength_name}支，库看{storage_name}支，空看{void_name}柱旬空，神煞看日干与日支"
    return time, basis


def _role_for(value: str, base: str) -> str | None:
    upper, lower = (BY_CODE[base[:3]].name, BY_CODE[base[3:]].name)
    compact = re.sub(r"[\s、,，；;为卦]", "", value)
    for role, opposite in (("体", "用"), ("用", "体")):
        if compact in {f"上{role}下{opposite}", f"下{opposite}上{role}"}:
            return role
        if upper != lower and compact in {
            f"{upper}{role}{lower}{opposite}",
            f"{lower}{opposite}{upper}{role}",
        }:
            return role
    return None


def prepare_reading(text: str) -> PreparedReading:
    fields, bare = _parse(text)
    method, platform = fields.get("方法", ""), fields.get("来源", "")
    source = method or platform
    declared_site = bool(SITE_SOURCE.fullmatch(source)) or source.lower() in {
        "iztro",
        "网站报数",
        "报数",
        "报数起卦",
    }
    other = bool(method or platform) and not declared_site
    trigrams = any(fields.get(key) for key in ("上卦", "下卦"))
    site_shape = all(
        fields.get(key) for key in ("时间", "四柱", "空亡", "本卦", "变卦")
    )
    given_charts = {layer: fields[layer] for layer in LAYERS if fields.get(layer)}
    moving = _moving_lines(fields["动爻"]) if fields.get("动爻") else None
    numbers, question = (
        ((), "") if other else _number_input(fields.get("报数", ""), bare)
    )
    if not (numbers or given_charts or trigrams):
        if other:
            raise InputError("该起卦法请直接提供本卦和变卦／动爻，然后重发完整命令。")
        raise InputError(MISSING)

    number_cast = None
    if numbers:
        try:
            number_cast = cast_numbers(numbers)
        except ValueError as exc:
            raise InputError(str(exc)) from exc

    entries = default_entries()
    given = {layer: _hexagram(value, entries) for layer, value in given_charts.items()}
    base = given.get("本卦")
    base_source = "用户提供"
    if trigrams:
        if not fields.get("上卦") or not fields.get("下卦"):
            raise InputError("请补齐上卦、下卦，再重发完整命令。")
        specified = _trigram_code(fields["上卦"]) + _trigram_code(fields["下卦"])
        if base is not None and base != specified:
            raise InputError("本卦与所给上卦、下卦不一致，请核对后重发。")
        if base is None:
            base, base_source = specified, "由用户指定的上下卦确定"
    if base is None and number_cast:
        base, base_source = number_cast.code, "由报数推导"
    if base is None:
        raise InputError(MISSING)

    mutual = mutual_code(base)
    if "互卦" in given and given["互卦"] != mutual:
        raise InputError(
            f"所给互卦与本卦推导的互卦「{entries[mutual].name}」不一致，请核对后重发。"
        )

    moving_source = "用户提供"
    if "变卦" in given:
        inferred = moving_between(base, given["变卦"])
        if moving is not None and moving != inferred:
            raise InputError("变卦与所给动爻不一致，请核对后重发。")
        if moving is None:
            moving, moving_source = inferred, "由本卦与变卦差异推导"
    if moving is None and number_cast:
        moving, moving_source = (number_cast.moving,), "由报数推导"
    if moving is None:
        if other or trigrams:
            raise InputError("请补充变卦或动爻，然后重发完整命令。")
        raise InputError("请补充变卦、动爻或报数，然后重发完整命令。")
    if number_cast and (base != number_cast.code or moving != (number_cast.moving,)):
        computed = entries[number_cast.code].name
        line = LINE_NAMES[number_cast.moving - 1]
        raise InputError(
            f"按网站报数得到「{computed}、{line}动」，与所给卦象不一致，请核对后重发。"
        )

    website = not other and (
        declared_site
        or (site_shape and not trigrams and len(moving) == 1)
        or (
            number_cast is not None
            and not given_charts
            and not trigrams
            and moving_source == "由报数推导"
        )
    )
    if website and len(moving) > 1:
        raise InputError(
            "网站报数只有一个动爻，当前排盘有多动爻，请核对来源或卦象后重发。"
        )
    time, time_basis = (
        _time_context(fields) if website else (None, "本次未纳入时间标签")
    )
    route = (
        f"用户指定（{method or platform}）"
        if other
        else "网站排盘规则"
        if website
        else "用户提供的卦象"
    )

    role_text = fields.get("体用", "")
    upper_role = _role_for(role_text, base) if role_text else None
    role_basis = f"用户提供：{role_text}" if role_text else "未判定"
    if len(moving) == 1:
        standard_role = "用" if moving[0] >= 4 else "体"
        if website and upper_role is not None and upper_role != standard_role:
            raise InputError("所给体用与网站动爻规则不一致，请核对来源或体用后重发。")
        if not role_text:
            upper_role = standard_role
            role_basis = "网站规则" if website else "本次假设：动爻所在经卦为用"

    codes = {"本卦": base, "互卦": mutual}
    if moving or "变卦" in given:
        codes["变卦"] = changed_code(base, moving)
    chart_sources = {
        "本卦": base_source,
        "互卦": "用户提供，已复核" if "互卦" in given else "由本卦推导",
        "变卦": "用户提供，已复核" if "变卦" in given else "由动爻推导",
    }
    lines = [f"采用规则：{route}"]
    if number_cast:
        lines.append(
            f"报数：{' '.join(map(str, numbers))} → 上卦 {number_cast.upper}、"
            f"下卦 {number_cast.lower}、动爻 {number_cast.moving}（余零按 8／6）"
        )
    if other and fields.get("报数"):
        lines.append("按用户指定的方法或卦象处理，未用网站报数重算。")
    line_names = "、".join(LINE_NAMES[line - 1] for line in moving) or "无动爻"
    lines.extend(
        [
            f"动爻：{line_names}（{moving_source}）",
            f"体用：{role_basis}",
            f"时间标签：{time_basis}",
        ]
    )
    charts = {}
    for layer, code in codes.items():
        chart = {"编码": code, "卦名": entries[code].name, "来源": chart_sources[layer]}
        for position, trigram, role in (
            ("上卦", BY_CODE[code[:3]], upper_role),
            ("下卦", BY_CODE[code[3:]], {"体": "用", "用": "体"}.get(upper_role)),
        ):
            parts = [trigram.name + trigram.element, *trigram_details(trigram, time)]
            if role:
                parts.insert(0, role)
            chart[position] = " ".join(parts)
        if website and upper_role:
            chart["生克"] = site_relation(
                BY_CODE[code[:3]], BY_CODE[code[3:]], upper_role, time
            )
        charts[layer] = chart
        relation = f"，{chart['生克']}" if "生克" in chart else ""
        lines.append(
            f"{layer}：{chart['卦名']} `{code}`（{chart['来源']}，"
            f"上 {chart['上卦']}／下 {chart['下卦']}{relation}）"
        )
    if "变卦" not in codes:
        lines.append("变卦：未提供（无动爻）")

    references = format_references(codes, entries)
    structured = {
        "占问": fields.get("占问") or question or ("" if numbers else bare),
        "采用规则": route,
        "用户提供的时间": fields.get("时间"),
        "用户提供的四柱": fields.get("四柱"),
        "用户提供的空亡": fields.get("空亡"),
        "时间标签基准": time_basis,
        "报数": list(numbers),
        "动爻": list(moving),
        "动爻来源": moving_source,
        "体用依据": role_basis,
        "卦象": charts,
    }
    model_input = (
        f"【本次输入】\n{text}\n\n"
        f"【结构化排盘结果】\n{json.dumps(structured, ensure_ascii=False, indent=2)}\n\n"
        f"【《灵签详解》检索原文】\n{references}"
    )
    return PreparedReading("\n".join(lines), references, model_input)
