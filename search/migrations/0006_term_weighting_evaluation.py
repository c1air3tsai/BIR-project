from django.db import migrations, models
import django.db.models.deletion


SEED_QUERIES = [
    'GLP-1 obesity', 'semaglutide weight loss', 'liraglutide diabetes',
    'GLP-1 cardiovascular', 'GLP-1 kidney disease', 'insulin secretion',
    'glucose control', 'appetite regulation', 'gastric emptying',
    'GLP-1 receptor agonist adverse effects',
]


def add_candidate_queries(apps, schema_editor):
    Query = apps.get_model('search', 'EvaluationQuery')
    for text in SEED_QUERIES:
        Query.objects.get_or_create(text=text)


class Migration(migrations.Migration):
    dependencies = [('search', '0005_document_arxiv_id_importjob_source')]
    operations = [
        migrations.CreateModel(name='EvaluationQuery', fields=[
            ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
            ('text', models.CharField(max_length=300, unique=True)),
            ('created_at', models.DateTimeField(auto_now_add=True)),
            ('updated_at', models.DateTimeField(auto_now=True)),
        ], options={'ordering': ['id']}),
        migrations.CreateModel(name='RelevanceJudgment', fields=[
            ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
            ('relevance', models.PositiveSmallIntegerField(choices=[(0, 'Not relevant'), (1, 'Relevant')])),
            ('source', models.CharField(default='manual', max_length=30)),
            ('updated_at', models.DateTimeField(auto_now=True)),
            ('document', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='relevance_judgments', to='search.document')),
            ('query', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='judgments', to='search.evaluationquery')),
        ]),
        migrations.AddConstraint(model_name='relevancejudgment', constraint=models.UniqueConstraint(fields=('query', 'document'), name='unique_relevance_judgment')),
        migrations.RunPython(add_candidate_queries, migrations.RunPython.noop),
    ]
