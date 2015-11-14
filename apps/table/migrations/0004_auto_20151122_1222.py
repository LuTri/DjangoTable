# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.db import models, migrations


class Migration(migrations.Migration):

    dependencies = [
        ('table', '0003_auto_20151121_0934'),
    ]

    operations = [
        migrations.AlterUniqueTogether(
            name='ledpos',
            unique_together=set([('table', 'pos')]),
        ),
    ]
