from django.urls import path
from . import views
from . import research_views

app_name = "search"

urlpatterns = [
    path("", views.search_view, name="home"),
    path("articles/", views.articles_view, name="articles"),
    path("document/<int:pk>/", views.document_detail_view, name="document_detail"),
    path("document/<int:pk>/xml/", views.download_xml_view, name="download_xml"),
    path("document/<int:pk>/delete/", views.delete_document_view, name="delete_document"),
    path("articles/delete-topic/", views.delete_topic_documents_view, name="delete_topic_documents"),
    path("zipf/", views.zipf_view, name="zipf"),
    path("zipf/export/", views.zipf_export_view, name="zipf_export"),
    path('analysis/domains/', research_views.domains_view, name='domains'),
    path('analysis/domains/export/', research_views.domains_export, name='domains_export'),
    path('analysis/word2vec/', research_views.word2vec_view, name='word2vec'),
    path('analysis/word2vec/export/', research_views.word2vec_export, name='word2vec_export'),
    path('analysis/report/', research_views.research_view, name='research'),
    path('analysis/report/export/', research_views.research_export, name='research_export'),
    path('analysis/matching/', research_views.distance_view, name='distance'),
    path("imports/<uuid:pk>/", views.job_status_view, name="job_status"),
    path("imports/<uuid:pk>/stop/", views.cancel_import_view, name="cancel_import"),
    path("upload/", views.import_view, name="import"),
]
