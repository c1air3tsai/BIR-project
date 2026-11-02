# search/templatetags/search_ui.py
# 模板小工具：query_url（改網址參數）、topic_query（保留主題篩選）。

from django import template
register = template.Library()


@register.simple_tag(takes_context=True)
def query_url(context, **changes):
    query = context['request'].GET.copy()
    for key, value in changes.items():
        if value is None:
            query.pop(key, None)
        else:
            query[key] = str(value)
    return '?' + query.urlencode()


@register.simple_tag(takes_context=True)
def topic_query(context):
    from urllib.parse import urlencode
    topics = context.get('selected_topics', [])
    return '?' + urlencode([('topic', value) for value in topics]) if topics else ''
