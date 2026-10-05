from django.db.models import Count
from .models import Document, Topic


def topic_context(request):
    selected = {value for value in request.GET.getlist('topic') if value.isdigit()}
    topics = list(Topic.objects.annotate(article_count=Count('documents', distinct=True)))
    valid = {str(topic.pk) for topic in topics}
    selected &= valid
    for topic in topics:
        topic.selected = str(topic.pk) in selected
    return {'topics': topics, 'selected_topics': sorted(selected), 'selected_topic_count': len(selected)}


def filter_documents(documents, selected_topics=None, year=''):
    if selected_topics:
        documents = documents.filter(topics__pk__in=selected_topics).distinct()
    if year:
        documents = documents.filter(publication_year=year)
    return documents
