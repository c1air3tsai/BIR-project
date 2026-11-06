# =============================================================================
# reuters/templatetags/reuters_ui.py
# Project 4 模板小工具：{% load reuters_ui %}
#   thousands  數字加上千分位逗號（9603 → 9,603）。網站語系是 zh-hant，Django 內建的 intcomma
#              會依中文習慣每 4 位分組，所以另外寫一個固定 3 位分組的版本
# =============================================================================
from django import template

register = template.Library()


@register.filter
def thousands(value):
    try:
        return f'{int(value):,}'
    except (TypeError, ValueError):
        return value
