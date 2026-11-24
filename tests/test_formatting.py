import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from astrbot_plugin_meihua.formatting import plain_text


class FormattingTests(unittest.TestCase):
    def test_screenshot_style_keeps_words_and_paragraphs(self):
        text = (
            "### 分层解读\n\n"
            "**本卦｜水泽节：可以试，先定好预算和边界。 **"
            "《灵签详解·水泽节》说“诸事必须节制，不宜过份”。\n\n"
            "**综合解读： **从有限范围开始。卦码 `010011`。\n\n"
            "1. 第一项\n2. 第二项"
        )
        self.assertEqual(
            plain_text(text),
            "分层解读\n\n本卦｜水泽节：可以试，先定好预算和边界。 "
            "《灵签详解·水泽节》说“诸事必须节制，不宜过份”。\n\n"
            "综合解读： 从有限范围开始。卦码 010011。\n\n1. 第一项\n2. 第二项",
        )

    def test_common_markup_preserves_literal_text_and_links(self):
        text = (
            "# 标题 #\n\n> 引文\n\n- **重点**\n+ *强调*\n* ~~旧文~~\n\n"
            "__bold__ _italic_ ***strong***\n\n---\n"
            "[参考](https://example.test/a_b?q=1)\n"
            "`provider_id` 3 * 4 = 12 snake_case a__b__c C#\n"
            "\\*保留星号\\*\n\n```text\n# 原样代码\nx = a * b\n```"
        )
        output = plain_text(text)
        self.assertIn("标题\n\n引文\n\n• 重点\n• 强调\n• 旧文", output)
        self.assertIn("bold italic strong", output)
        self.assertIn("参考（https://example.test/a_b?q=1）", output)
        self.assertIn("provider_id 3 * 4 = 12 snake_case a__b__c C#", output)
        self.assertIn("*保留星号*", output)
        self.assertTrue(output.endswith("# 原样代码\nx = a * b"))
        self.assertNotIn("```", output)


if __name__ == "__main__":
    unittest.main()
