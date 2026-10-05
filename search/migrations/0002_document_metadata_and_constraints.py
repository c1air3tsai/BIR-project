from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [
        ("search", "0001_initial"),
    ]

    operations = [
        migrations.AddField(
            model_name="document",
            name="pmcid",
            field=models.CharField(blank=True, db_index=True, max_length=40),
        ),
        migrations.AddField(
            model_name="document",
            name="doi",
            field=models.CharField(blank=True, max_length=200),
        ),
        migrations.AddField(
            model_name="document",
            name="journal",
            field=models.CharField(blank=True, max_length=300),
        ),
        migrations.AddField(
            model_name="document",
            name="publication_year",
            field=models.CharField(blank=True, max_length=10),
        ),
        migrations.AddField(
            model_name="document",
            name="authors",
            field=models.TextField(blank=True),
        ),
        migrations.AddField(
            model_name="document",
            name="abstract",
            field=models.TextField(blank=True),
        ),
        migrations.AddField(
            model_name="document",
            name="avg_words_per_sentence",
            field=models.FloatField(default=0),
        ),
        migrations.AlterField(
            model_name="document",
            name="source_file",
            field=models.CharField(max_length=300, unique=True),
        ),
        migrations.AlterField(
            model_name="document",
            name="indexed_at",
            field=models.DateTimeField(auto_now=True),
        ),
        migrations.AlterModelOptions(
            name="document",
            options={"ordering": ["-indexed_at", "title"]},
        ),
        migrations.AlterUniqueTogether(
            name="posting",
            unique_together=set(),
        ),
        migrations.AddConstraint(
            model_name="posting",
            constraint=models.UniqueConstraint(fields=("term", "document"), name="unique_term_document"),
        ),
    ]
