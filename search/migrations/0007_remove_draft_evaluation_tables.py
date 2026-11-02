# 移除 Project 3 草稿版的人工判定資料表。
# Project 3 改用 weighting/ app，相關性改由 MeSH 規則自動產生（見 weighting/queries.py），
# 不再需要 EvaluationQuery / RelevanceJudgment 兩張表。
from django.db import migrations


class Migration(migrations.Migration):
    dependencies = [('search', '0006_term_weighting_evaluation')]
    operations = [
        migrations.DeleteModel(name='RelevanceJudgment'),
        migrations.DeleteModel(name='EvaluationQuery'),
    ]
