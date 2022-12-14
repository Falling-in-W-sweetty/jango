# Generated by Django 3.2.10 on 2022-07-18 03:48

from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='Bowendu',
            fields=[
                ('report_id', models.IntegerField(db_column='REPORT_ID', primary_key=True, serialize=False)),
                ('time', models.DateTimeField(db_column='TIME')),
                ('hd', models.FloatField(db_column='HD')),
                ('hd10', models.FloatField(db_column='HD10')),
                ('lyl', models.FloatField(db_column='LYL')),
                ('cz', models.FloatField(db_column='CZ')),
                ('lyl10', models.FloatField(db_column='LYL10')),
                ('zcd', models.FloatField(db_column='ZCD')),
                ('s1', models.FloatField(db_column='S1')),
                ('s2', models.FloatField(db_column='S2')),
                ('s3', models.FloatField(db_column='S3')),
                ('m1', models.CharField(db_column='M1', max_length=11)),
                ('m2', models.FloatField(db_column='M2')),
                ('m3', models.FloatField(db_column='M3')),
                ('n1', models.FloatField(db_column='N1')),
                ('n2', models.FloatField(db_column='N2')),
                ('n3', models.FloatField(db_column='N3')),
                ('ave', models.FloatField(db_column='AVE')),
                ('min', models.FloatField(db_column='MIN')),
                ('max', models.FloatField(db_column='MAX')),
                ('cbdgs', models.FloatField(db_column='CBDGS')),
                ('s_max', models.FloatField(db_column='S-MAX')),
                ('m_max', models.FloatField(db_column='M-MAX')),
                ('n_max', models.FloatField(db_column='N-MAX')),
                ('fcz', models.FloatField(db_column='FCZ')),
            ],
            options={
                'db_table': 'bowendu',
                'managed': True,
            },
        ),
        migrations.CreateModel(
            name='Houdu',
            fields=[
                ('id', models.IntegerField(db_column='REPORT_ID', primary_key=True, serialize=False)),
                ('time', models.DateTimeField(db_column='TIME')),
                ('myhd', models.CharField(db_column='MYHD', max_length=255)),
                ('bk', models.CharField(db_column='BK', max_length=255)),
                ('yj', models.CharField(db_column='YJ', max_length=255)),
                ('jbk', models.CharField(db_column='JBK', max_length=255)),
                ('s1', models.CharField(db_column='S1', max_length=255)),
                ('s2', models.CharField(db_column='S2', max_length=255)),
                ('s3', models.CharField(db_column='S3', max_length=255)),
                ('s4', models.CharField(db_column='S4', max_length=255)),
                ('s5', models.CharField(db_column='S5', max_length=255)),
                ('m1', models.CharField(db_column='M1', max_length=255)),
                ('m2', models.CharField(db_column='M2', max_length=255)),
                ('m3', models.CharField(db_column='M3', max_length=255)),
                ('m4', models.CharField(db_column='M4', max_length=255)),
                ('m5', models.CharField(db_column='M5', max_length=255)),
                ('n1', models.CharField(db_column='N1', max_length=255)),
                ('n2', models.CharField(db_column='N2', max_length=255)),
                ('n3', models.CharField(db_column='N3', max_length=255)),
                ('n4', models.CharField(db_column='N4', max_length=255)),
                ('n5', models.CharField(db_column='N5', max_length=255)),
                ('ave', models.CharField(db_column='AVE', max_length=255)),
                ('min', models.CharField(db_column='MIN', max_length=255)),
                ('max', models.CharField(db_column='MAX', max_length=255)),
                ('hbc', models.CharField(db_column='HBC', max_length=255)),
                ('bhn', models.CharField(db_column='BHN', max_length=255)),
                ('bhb', models.CharField(db_column='BHB', max_length=255)),
                ('bks', models.CharField(db_column='BKS', max_length=255)),
                ('bkn', models.CharField(db_column='BKN', max_length=255)),
            ],
            options={
                'db_table': 'houdu',
                'managed': True,
            },
        ),
        migrations.CreateModel(
            name='user',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('value1', models.CharField(max_length=32)),
                ('value2', models.CharField(max_length=32)),
                ('value3', models.CharField(max_length=32)),
            ],
            options={
                'db_table': 'ulriv',
            },
        ),
    ]
