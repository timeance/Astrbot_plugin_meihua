from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

MAX_SAFE_INTEGER = 9_007_199_254_740_991
STEMS = "甲乙丙丁戊己庚辛壬癸"
BRANCHES = "子丑寅卯辰巳午未申酉戌亥"
LINE_NAMES = ("初爻", "二爻", "三爻", "四爻", "五爻", "上爻")


class Trigram(NamedTuple):
    name: str
    code: str
    element: str
    branches: str
    storage: str
    void: str
    relations: tuple[str, str, str, str]


# Keep the site's trigram order, tag constants, and directed relation table.
TRIGRAMS = (
    Trigram("乾", "111", "金", "戌亥", "丑", "戌亥", ("坎", "震巽", "乾", "兑")),
    Trigram("兑", "011", "金", "酉", "丑", "申酉", ("坎", "震巽", "兑", "乾")),
    Trigram("离", "101", "火", "午", "戌", "午未", ("艮坤", "乾兑", "离", "")),
    Trigram("震", "001", "木", "卯", "未", "寅卯", ("离", "艮坤", "震", "巽")),
    Trigram("巽", "110", "木", "辰巳", "未", "辰巳", ("离", "艮坤", "巽", "震")),
    Trigram("坎", "010", "水", "子", "辰", "子丑", ("震巽", "离", "坎", "")),
    Trigram("艮", "100", "土", "丑寅", "", "子丑", ("乾兑", "坎", "艮坤", "")),
    Trigram("坤", "000", "土", "未申", "", "午未", ("乾兑", "坎", "艮坤", "")),
)
BY_CODE = {gua.code: gua for gua in TRIGRAMS}
RILU = dict(zip(STEMS, "寅卯巳午巳午申酉亥子"))
GUIREN = dict(
    zip(
        STEMS,
        (
            "丑未",
            "子申",
            "亥酉",
            "亥酉",
            "丑未",
            "子申",
            "丑未",
            "午寅",
            "巳卯",
            "巳卯",
        ),
    )
)
TAOHUA = dict(zip(BRANCHES, "酉午卯子酉午卯子酉午卯子"))
YIMA = dict(zip(BRANCHES, "寅亥申巳寅亥申巳寅亥申巳"))
WANG_ORDER = {
    branch: order
    for branches, order in (
        ("寅卯", "木火水金土"),
        ("辰未戌丑", "土金火木水"),
        ("巳午", "火土木水金"),
        ("申酉", "金水土火木"),
        ("亥子", "水木金土火"),
    )
    for branch in branches
}


@dataclass(frozen=True)
class NumberCast:
    upper: int
    lower: int
    moving: int

    @property
    def code(self) -> str:
        return TRIGRAMS[self.upper - 1].code + TRIGRAMS[self.lower - 1].code


@dataclass(frozen=True)
class SiteTime:
    strength_branch: str
    storage_branch: str
    day_pillar: str
    selected_void: str


def cast_numbers(numbers: tuple[int, ...]) -> NumberCast:
    if not numbers or min(numbers) <= 0:
        raise ValueError("报数须为正整数，请补齐后重发。")
    total = sum(numbers)
    if total > MAX_SAFE_INTEGER:
        raise ValueError("报数总和超出网站的安全整数范围，请改用较小的正整数后重发。")
    if len(numbers) == 1:
        upper = lower = total
    else:
        upper = sum(numbers[: len(numbers) // 2])
        lower = total - upper
    return NumberCast(upper % 8 or 8, lower % 8 or 8, total % 6 or 6)


def mutual_code(code: str) -> str:
    return code[1:4] + code[2:5]


def changed_code(code: str, moving: tuple[int, ...]) -> str:
    bits = list(code)
    for line in moving:
        position = 6 - line
        bits[position] = "0" if bits[position] == "1" else "1"
    return "".join(bits)


def moving_between(base: str, changed: str) -> tuple[int, ...]:
    return tuple(line for line in range(1, 7) if base[6 - line] != changed[6 - line])


def trigram_details(gua: Trigram, time: SiteTime | None) -> tuple[str, ...]:
    if time is None:
        return ()
    stem, branch = time.day_pillar
    return (trigram_state(gua, time),) + tuple(
        tag
        for tag, matches in (
            ("日禄", RILU[stem] in gua.branches),
            ("驿马", YIMA[branch] in gua.branches),
            ("桃花", TAOHUA[branch] in gua.branches),
            ("贵人", any(item in gua.branches for item in GUIREN[stem])),
            ("库", gua.storage == time.storage_branch),
            ("空", any(item in gua.void for item in time.selected_void)),
        )
        if matches
    )


def trigram_state(gua: Trigram, time: SiteTime | None) -> str:
    return (
        "旺相休囚死"[WANG_ORDER[time.strength_branch].index(gua.element)]
        if time
        else ""
    )


def site_relation(
    upper: Trigram, lower: Trigram, upper_role: str, time: SiteTime | None
) -> str:
    ti, yong = (upper, lower) if upper_role == "体" else (lower, upper)
    for subject, obj, subject_role, object_role in (
        (ti, yong, "体", "用"),
        (yong, ti, "用", "体"),
    ):
        for symbol, targets in zip("生克冲合", subject.relations):
            if obj.name in targets:
                return f"{trigram_state(subject, time)}{subject_role}{symbol}{trigram_state(obj, time)}{object_role}"
    raise ValueError("网站经卦关系表缺少对应关系。")
