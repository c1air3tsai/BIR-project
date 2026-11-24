# =============================================================================
# indexing/templatetags/indexing_ui.py
# Project 5 頁面用的小工具：{% load indexing_ui %}
#   {% doc_buttons ids terms %}   一串 docID → 可以點的小方塊；點了開彈出視窗看全文，terms 用螢光筆標出來
#                                 （ids 可以是 list 或空白分隔的字串；彈出視窗的程式在 static/indexing/indexing.js）
# =============================================================================
from django import template
from django.utils.html import format_html_join

register = template.Library()


@register.simple_tag
def doc_buttons(ids, terms):
    if isinstance(ids, str):
        ids = ids.split()
    return format_html_join('', '<button type="button" class="ix-doc" data-doc="{0}" data-terms="{1}" '
                                'title="Doc {0}：看全文">{0}</button>', ((doc_id, terms) for doc_id in ids))
