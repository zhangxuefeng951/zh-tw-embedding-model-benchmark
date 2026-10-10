"""前綴（Prefix）樣板。

樣板可以使用：
- {title}：文件標題，沒有標題時代入 none（EmbeddingGemma 的官方寫法）
- {text}：原始文字；樣板裡沒有 {text} 時，文字直接接在樣板後面（一般前綴的用法）

例如 EmbeddingGemma 2 的文件前綴 "title: {title} | text: "。
"""

from __future__ import annotations

import re

_TOKEN = re.compile(r"\{(title|text)\}")


def render(template: str | None, text: str, title: str | None = None) -> str:
    template = template or ""
    title_value = title.strip() if title and title.strip() else "none"
    has_text = "{text}" in template

    def sub(m: re.Match) -> str:
        return title_value if m.group(1) == "title" else text

    rendered = _TOKEN.sub(sub, template)
    return rendered if has_text else rendered + text


def uses_title(template: str | None) -> bool:
    return bool(template) and "{title}" in template


def display(template: str | None) -> str:
    """在介面與報告中顯示前綴：讓換行與尾端空白看得見。"""
    if not template:
        return "（無前綴）"
    return template.replace("\n", "\\n")
