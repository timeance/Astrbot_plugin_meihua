import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from astrbot_plugin_meihua.core import InputError, prepare_reading
from astrbot_plugin_meihua.knowledge import EXPECTED_FIELDS, REFERENCE

SITE_SAMPLE = """占问: 尝试小红书推广
时间: 2026年11月22日 17(酉)时25分(十月十四)
四柱: 丙午 己亥 庚子 乙酉
本卦: 水泽节
变卦: 坎为水
空亡: 寅卯 辰巳 辰巳 午未"""

CLIPBOARD_SAMPLE = """事由\t
合成测试事项
报数\t11 12
时间\t2026/11/24 00:09:29
节气\t小雪 2026-11-22 15:23:18
大雪 2026-12-07 10:52:29
时间\t年\t月\t日\t时
丙午\t己亥\t壬寅\t庚子
空亡\t寅卯\t辰巳\t辰巳\t辰巳
☳震\t☴巽\t☴巽\t☴巽
&#x20;
用
&#x20;
体
&#x20;
本卦
火雷噬嗑
桃花
贵人
相体
生
死用
互卦
水山蹇
囚体
克
旺用
变卦
天雷无妄
日禄
桃花
贵人
休用
克
相体"""


def chart_data(reading):
    return json.loads(
        reading.model_input.split("【结构化排盘结果】\n", 1)[1].split(
            "\n\n【《灵签详解》检索原文】", 1
        )[0]
    )


class CoreTests(unittest.TestCase):
    def test_website_clipboard_table_and_wrapped_cells(self):
        variants = (
            CLIPBOARD_SAMPLE,
            CLIPBOARD_SAMPLE.replace("\n", "\\\n"),
            CLIPBOARD_SAMPLE.replace("\t", "\n"),
            CLIPBOARD_SAMPLE.replace("\t", " ").replace("&#x20;", "&nbsp;"),
        )
        for text in variants:
            with self.subTest(text=text):
                reading = prepare_reading(text)
                data = chart_data(reading)
                self.assertIn(text, reading.model_input)
                self.assertEqual(data["占问"], "合成测试事项")
                self.assertEqual(data["报数"], [11, 12])
                self.assertEqual(data["动爻"], [5])
                self.assertEqual(data["采用规则"], "网站排盘规则")
                self.assertEqual(data["用户提供的时间"], "2026/11/24 00:09:29")
                self.assertEqual(data["用户提供的四柱"], "丙午 己亥 壬寅 庚子")
                self.assertEqual(data["用户提供的空亡"], "寅卯 辰巳 辰巳 辰巳")
                self.assertEqual(
                    [layer["编码"] for layer in data["卦象"].values()],
                    ["101001", "010100", "111001"],
                )
                self.assertIn("网站默认", data["时间标签基准"])
                self.assertIn("用 离火 死", data["卦象"]["本卦"]["上卦"])
                self.assertIn("体 震木 相", data["卦象"]["本卦"]["下卦"])
                self.assertEqual(reading.references.count("对应层次："), 3)

    def test_clipboard_missing_values_and_conflicts_stay_explicit(self):
        text = CLIPBOARD_SAMPLE.replace("报数\t11 12\n", "")
        self.assertEqual(chart_data(prepare_reading(text))["动爻"], [5])
        for text, message in (
            ("事由\n合成事项 3000 元\n时间\t2026/11/24 00:09:29", "补充报数"),
            ("事由\n报数\n11 12\n本卦\n变卦\n天雷无妄", "本卦.*缺少"),
            (CLIPBOARD_SAMPLE + "\n本卦\n乾为天", "重复且不一致"),
            (CLIPBOARD_SAMPLE.replace("水山蹇", "乾为天"), "所给互卦"),
            (CLIPBOARD_SAMPLE + "\n时间\t2026/11/25 00:09:29", "重复且不一致"),
        ):
            with self.subTest(text=text), self.assertRaisesRegex(InputError, message):
                prepare_reading(text)

    def test_site_sample_with_fence_and_chinese_colon(self):
        text = "```makefile\n" + SITE_SAMPLE.replace(": ", "：") + "\n```"
        reading = prepare_reading(text)
        data = chart_data(reading)
        self.assertIn(text, reading.model_input)
        self.assertEqual(data["占问"], "尝试小红书推广")
        self.assertEqual(data["动爻"], [1])
        self.assertEqual(data["报数"], [])
        self.assertEqual(
            [layer["编码"] for layer in data["卦象"].values()],
            ["010011", "100001", "010010"],
        )
        self.assertIn("动爻：初爻（由本卦与变卦差异推导）", reading.summary)
        self.assertIn("体 坎水 旺", reading.summary)
        self.assertIn("用 兑金 休 桃花", reading.summary)
        self.assertIn("旺体", reading.summary)

    def test_labeled_and_bare_numbers_match(self):
        for text in (
            "报数:11 12\n事由: 2026年推广预算 3000 元",
            "11 12 2026年推广预算 3000 元",
            "11 12 占问：2026年推广预算 3000 元",
        ):
            with self.subTest(text=text):
                # The explicitly labeled question sets the boundary when it starts with a number.
                if text.startswith("11 12 2026"):
                    with self.assertRaises(InputError):
                        prepare_reading(text)
                    continue
                reading = prepare_reading(text)
                data = chart_data(reading)
                self.assertEqual(data["占问"], "2026年推广预算 3000 元")
                self.assertEqual(data["报数"], [11, 12])
                self.assertEqual(data["动爻"], [5])
                self.assertEqual(
                    [layer["编码"] for layer in data["卦象"].values()],
                    ["101001", "010100", "111001"],
                )
                self.assertIn("本次未纳入时间标签", reading.summary)
        self.assertEqual(
            chart_data(prepare_reading("11 12 问题怎么处理"))["报数"], [11, 12]
        )
        self.assertEqual(
            chart_data(prepare_reading("报数: 3 4 5"))["卦象"]["本卦"]["编码"], "101111"
        )

    def test_dates_and_money_are_not_casting_numbers(self):
        for text in (
            "占问：2026 年要不要投入 3000 元？\n时间：2026年11月22日 17时25分",
            "2026 年 11 月 22 日适合发布吗？",
            "3000 元预算怎么分？",
            "3000元怎么花？",
        ):
            with (
                self.subTest(text=text),
                self.assertRaisesRegex(InputError, "补充报数"),
            ):
                prepare_reading(text)

    def test_ambiguous_and_invalid_numbers(self):
        with self.assertRaisesRegex(InputError, "六位卦码，还是一个报数"):
            prepare_reading("101001")
        for value in ("-11 12", "11.5 12", "11 0"):
            with (
                self.subTest(value=value),
                self.assertRaisesRegex(InputError, "正整数"),
            ):
                prepare_reading(f"报数：{value}")
        self.assertEqual(
            chart_data(prepare_reading("卦码:101001\n动爻:五"))["报数"], []
        )
        self.assertEqual(chart_data(prepare_reading("报数:101001"))["报数"], [101001])

    def test_conflicts_raise_instead_of_replacing_explicit_results(self):
        for text, message in (
            (SITE_SAMPLE + "\n互卦: 乾为天", "所给互卦"),
            (SITE_SAMPLE + "\n动爻: 二爻", "所给动爻"),
            (SITE_SAMPLE + "\n报数: 11 12", "按网站报数"),
            ("卦码: 水泽节 (101001)\n动爻: 初爻", "卦名与六位卦码"),
            ("本卦: 水泽节", "补充变卦"),
            ("方法: 时间起卦\n报数: 11 12", "直接提供本卦"),
        ):
            with self.subTest(text=text), self.assertRaisesRegex(InputError, message):
                prepare_reading(text)

    def test_manual_multiple_lines_and_single_line_assumption(self):
        text = "方法:铜钱法\n来源:https://meihua.iztro.com/\n报数:11 12\n" + SITE_SAMPLE
        text = text.replace("坎为水", "地水师")
        reading = prepare_reading(text)
        data = chart_data(reading)
        self.assertEqual(data["动爻"], [1, 5])
        self.assertEqual(data["报数"], [])
        self.assertEqual(data["卦象"]["本卦"]["上卦"], "坎水")
        self.assertEqual(data["体用依据"], "未判定")
        self.assertIn("本次未纳入时间标签", reading.summary)
        for text in ("上卦取 3、下卦取 4、五爻动", "上卦 取 3、下卦 取 4、五爻动"):
            reading = prepare_reading(text)
            self.assertIn("假设：动爻所在经卦为用", reading.summary)
            self.assertIn("火雷噬嗑 `101001`", reading.summary)

    def test_references_preserve_exact_fields_and_deduplicate(self):
        reading = prepare_reading("本卦：乾为天\n动爻：初爻")
        blocks = reading.references.split("\n\n")
        self.assertEqual(len(blocks), 2)
        self.assertTrue(
            blocks[0].startswith(
                "《灵签详解》｜第1卦 乾为天｜编码111111\n对应层次：本卦、互卦\n"
            )
        )
        self.assertIn("对应层次：变卦", blocks[1])
        source = REFERENCE.read_text(encoding="utf-8-sig")
        for block in blocks:
            _, _, fields = block.split("\n", 2)
            self.assertIn(f"\n{fields}\n", source)
            self.assertEqual(fields.count("\n") + 1, len(EXPECTED_FIELDS))
            for name, line in zip(EXPECTED_FIELDS, fields.splitlines()):
                self.assertTrue(line.startswith(f"- {name}："), line)
        self.assertTrue(reading.model_input.endswith(reading.references))


if __name__ == "__main__":
    unittest.main()
