from django.db import models
from django.db import models


class user(models.Model):  # 现在该模型在migtations中已经注册了，
    # 因此进行迁移的时候不会重复注册 注册文件内 有module.uer和数据库中table的对应关系
    # 如果没有对应关系，那么进行迁移的时候 才会重新创建表，每一次进行迁移时的操作都在迁移文件当中
    # report_id = models.IntegerField(db_column='REPORT_ID', primary_key=True)  # Field name made lowercase.
    value1 = models.CharField(max_length=32, null=False)
    value2 = models.CharField(max_length=32, null=False)
    value3 = models.CharField(max_length=32, null=False)

    class Meta:
        db_table = "ulriv"  # 这里可以更改数据库中的表名
        app_label = "datadb"

##添加表字段以及修改表名称为user
# from django.db import models


class Houdu(models.Model):
    id = models.IntegerField(db_column='REPORT_ID', primary_key=True)  # Field name made lowercase.
    time = models.DateTimeField(db_column='TIME')  # Field name made lowercase.
    myhd = models.CharField(db_column='MYHD', max_length=255)  # Field name made lowercase.
    bk = models.CharField(db_column='BK', max_length=255)  # Field name made lowercase.
    yj = models.CharField(db_column='YJ', max_length=255)  # Field name made lowercase.
    jbk = models.CharField(db_column='JBK', max_length=255)  # Field name made lowercase.
    s1 = models.CharField(db_column='S1', max_length=255)  # Field name made lowercase.
    s2 = models.CharField(db_column='S2', max_length=255)  # Field name made lowercase.
    s3 = models.CharField(db_column='S3', max_length=255)  # Field name made lowercase.
    s4 = models.CharField(db_column='S4', max_length=255)  # Field name made lowercase.
    s5 = models.CharField(db_column='S5', max_length=255)  # Field name made lowercase.
    m1 = models.CharField(db_column='M1', max_length=255)  # Field name made lowercase.
    m2 = models.CharField(db_column='M2', max_length=255)  # Field name made lowercase.
    m3 = models.CharField(db_column='M3', max_length=255)  # Field name made lowercase.
    m4 = models.CharField(db_column='M4', max_length=255)  # Field name made lowercase.
    m5 = models.CharField(db_column='M5', max_length=255)  # Field name made lowercase.
    n1 = models.CharField(db_column='N1', max_length=255)  # Field name made lowercase.
    n2 = models.CharField(db_column='N2', max_length=255)  # Field name made lowercase.
    n3 = models.CharField(db_column='N3', max_length=255)  # Field name made lowercase.
    n4 = models.CharField(db_column='N4', max_length=255)  # Field name made lowercase.
    n5 = models.CharField(db_column='N5', max_length=255)  # Field name made lowercase.
    ave = models.CharField(db_column='AVE', max_length=255)  # Field name made lowercase.
    min = models.CharField(db_column='MIN', max_length=255)  # Field name made lowercase.
    max = models.CharField(db_column='MAX', max_length=255)  # Field name made lowercase.
    hbc = models.CharField(db_column='HBC', max_length=255)  # Field name made lowercase.
    bhn = models.CharField(db_column='BHN', max_length=255)  # Field name made lowercase.
    bhb = models.CharField(db_column='BHB', max_length=255)  # Field name made lowercase.
    bks = models.CharField(db_column='BKS', max_length=255)  # Field name made lowercase.
    bkn = models.CharField(db_column='BKN', max_length=255)  # Field name made lowercase.

    class Meta:
        managed = True
        db_table = 'houdu'
        app_label = "datadb"


# Create your models here.
class Bowendu(models.Model):
    report_id = models.IntegerField(db_column='REPORT_ID', primary_key=True)  # Field name made lowercase.
    time = models.DateTimeField(db_column='TIME')  # Field name made lowercase.
    hd = models.FloatField(db_column='HD')  # Field name made lowercase.
    hd10 = models.FloatField(db_column='HD10')  # Field name made lowercase.
    lyl = models.FloatField(db_column='LYL')  # Field name made lowercase.
    cz = models.FloatField(db_column='CZ')  # Field name made lowercase.
    lyl10 = models.FloatField(db_column='LYL10')  # Field name made lowercase.
    zcd = models.FloatField(db_column='ZCD')  # Field name made lowercase.
    s1 = models.FloatField(db_column='S1')  # Field name made lowercase.
    s2 = models.FloatField(db_column='S2')  # Field name made lowercase.
    s3 = models.FloatField(db_column='S3')  # Field name made lowercase.
    m1 = models.CharField(db_column='M1', max_length=11)  # Field name made lowercase.
    m2 = models.FloatField(db_column='M2')  # Field name made lowercase.
    m3 = models.FloatField(db_column='M3')  # Field name made lowercase.
    n1 = models.FloatField(db_column='N1')  # Field name made lowercase.
    n2 = models.FloatField(db_column='N2')  # Field name made lowercase.
    n3 = models.FloatField(db_column='N3')  # Field name made lowercase.
    ave = models.FloatField(db_column='AVE')  # Field name made lowercase.
    min = models.FloatField(db_column='MIN')  # Field name made lowercase.
    max = models.FloatField(db_column='MAX')  # Field name made lowercase.
    cbdgs = models.FloatField(db_column='CBDGS')  # Field name made lowercase.
    s_max = models.FloatField(
        db_column='S-MAX')  # Field name made lowercase. Field renamed to remove unsuitable characters.
    m_max = models.FloatField(
        db_column='M-MAX')  # Field name made lowercase. Field renamed to remove unsuitable characters.
    n_max = models.FloatField(
        db_column='N-MAX')  # Field name made lowercase. Field renamed to remove unsuitable characters.
    fcz = models.FloatField(db_column='FCZ')  # Field name made lowercase.

    class Meta:
        managed = True
        db_table = 'bowendu'
        app_label = "datadb"
