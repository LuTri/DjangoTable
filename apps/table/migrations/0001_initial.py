# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.db import models, migrations


class Migration(migrations.Migration):

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='Color',
            fields=[
                ('id', models.AutoField(verbose_name='ID', serialize=False, auto_created=True, primary_key=True)),
                ('r', models.IntegerField(default=0)),
                ('g', models.IntegerField(default=0)),
                ('b', models.IntegerField(default=0)),
            ],
            options={
            },
            bases=(models.Model,),
        ),
        migrations.CreateModel(
            name='LedPos',
            fields=[
                ('id', models.AutoField(verbose_name='ID', serialize=False, auto_created=True, primary_key=True)),
                ('pos', models.CharField(max_length=5, choices=[(0, b'0_0'), (1, b'0_1'), (2, b'0_2'), (3, b'0_3'), (4, b'0_4'), (5, b'0_5'), (6, b'0_6'), (7, b'0_7'), (8, b'0_8'), (9, b'0_9'), (10, b'0_10'), (11, b'0_11'), (12, b'0_12'), (13, b'0_13'), (14, b'0_14'), (15, b'0_15'), (16, b'1_0'), (17, b'1_1'), (18, b'1_2'), (19, b'1_3'), (20, b'1_4'), (21, b'1_5'), (22, b'1_6'), (23, b'1_7'), (24, b'1_8'), (25, b'1_9'), (26, b'1_10'), (27, b'1_11'), (28, b'1_12'), (29, b'1_13'), (30, b'1_14'), (31, b'1_15'), (32, b'2_0'), (33, b'2_1'), (34, b'2_2'), (35, b'2_3'), (36, b'2_4'), (37, b'2_5'), (38, b'2_6'), (39, b'2_7'), (40, b'2_8'), (41, b'2_9'), (42, b'2_10'), (43, b'2_11'), (44, b'2_12'), (45, b'2_13'), (46, b'2_14'), (47, b'2_15'), (48, b'3_0'), (49, b'3_1'), (50, b'3_2'), (51, b'3_3'), (52, b'3_4'), (53, b'3_5'), (54, b'3_6'), (55, b'3_7'), (56, b'3_8'), (57, b'3_9'), (58, b'3_10'), (59, b'3_11'), (60, b'3_12'), (61, b'3_13'), (62, b'3_14'), (63, b'3_15'), (64, b'4_0'), (65, b'4_1'), (66, b'4_2'), (67, b'4_3'), (68, b'4_4'), (69, b'4_5'), (70, b'4_6'), (71, b'4_7'), (72, b'4_8'), (73, b'4_9'), (74, b'4_10'), (75, b'4_11'), (76, b'4_12'), (77, b'4_13'), (78, b'4_14'), (79, b'4_15'), (80, b'5_0'), (81, b'5_1'), (82, b'5_2'), (83, b'5_3'), (84, b'5_4'), (85, b'5_5'), (86, b'5_6'), (87, b'5_7'), (88, b'5_8'), (89, b'5_9'), (90, b'5_10'), (91, b'5_11'), (92, b'5_12'), (93, b'5_13'), (94, b'5_14'), (95, b'5_15'), (96, b'6_0'), (97, b'6_1'), (98, b'6_2'), (99, b'6_3'), (100, b'6_4'), (101, b'6_5'), (102, b'6_6'), (103, b'6_7'), (104, b'6_8'), (105, b'6_9'), (106, b'6_10'), (107, b'6_11'), (108, b'6_12'), (109, b'6_13'), (110, b'6_14'), (111, b'6_15')])),
                ('color', models.ForeignKey(to='table.Color')),
            ],
            options={
            },
            bases=(models.Model,),
        ),
        migrations.CreateModel(
            name='Table',
            fields=[
                ('id', models.AutoField(verbose_name='ID', serialize=False, auto_created=True, primary_key=True)),
                ('description', models.CharField(max_length=200, verbose_name=b'ColorScheme')),
            ],
            options={
            },
            bases=(models.Model,),
        ),
        migrations.AddField(
            model_name='ledpos',
            name='table',
            field=models.ForeignKey(to='table.Table'),
            preserve_default=True,
        ),
    ]
