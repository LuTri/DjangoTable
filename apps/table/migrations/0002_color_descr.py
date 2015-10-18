# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.db import models, migrations


class Migration(migrations.Migration):

    dependencies = [
        ('table', '0001_initial'),
    ]

    operations = [
        migrations.AddField(
            model_name='color',
            name='descr',
            field=models.CharField(default=b'', max_length=200, verbose_name=b'Color "Name"'),
            preserve_default=True,
        ),
    ]
