from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [
        ("search", "0002_document_metadata_and_constraints"),
    ]

    operations = [
        migrations.AddField(
            model_name="document",
            name="pmid",
            field=models.CharField(blank=True, db_index=True, max_length=40),
        ),
    ]
