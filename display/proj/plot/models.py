# This is an auto-generated Django model module.
# You'll have to do the following manually to clean this up:
#   * Rearrange models' order
#   * Make sure each model has one field with primary_key=True
#   * Make sure each ForeignKey has `on_delete` set to the desired behavior.
#   * Remove `managed = False` lines if you wish to allow Django to create, modify, and delete the table
# Feel free to rename the models, but don't rename db_table values or field names.
from __future__ import unicode_literals

from django.db import models


class AlgorithmRgb(models.Model):
#    id = models.IntegerField(primary_key=True)  # AutoField?
    R1 = models.IntegerField(db_column='R1')  # Field name made lowercase.
    R1C = models.IntegerField(db_column='R1C')  # Field name made lowercase.
    R2 = models.IntegerField(db_column='R2')  # Field name made lowercase.
    R2C = models.IntegerField(db_column='R2C')  # Field name made lowercase.
    G1 = models.IntegerField(db_column='G1')  # Field name made lowercase.
    G1C = models.IntegerField(db_column='G1C')  # Field name made lowercase.
    G2 = models.IntegerField(db_column='G2')  # Field name made lowercase.
    G2C = models.IntegerField(db_column='G2C')  # Field name made lowercase.
    B1 = models.IntegerField(db_column='B1')  # Field name made lowercase.
    B1C = models.IntegerField(db_column='B1C')  # Field name made lowercase.
    B2 = models.IntegerField(db_column='B2')  # Field name made lowercase.
    B2C = models.IntegerField(db_column='B2C')  # Field name made lowercase.
    Datetime = models.DateTimeField(db_column='Datetime')  # Field name made lowercase.
    Statue1 = models.IntegerField(db_column='Statue1')  # Field name made lowercase.
    Statue2 = models.IntegerField(db_column='Statue2')  # Field name made lowercase.
    Statue3 = models.IntegerField(db_column='Statue3')  # Field name made lowercase.
    Timestamp = models.IntegerField(db_column='Timestamp')  # Field name made lowercase.
    Uid = models.CharField(db_column='Uid', max_length=30)  # Field name made lowercase.
    def __unicode__(self):   
        return u'%s--- %s---%d-%d-%d-%d-%d-%d-%d-%d-%d-%d-%d-%d' % (self.Datetime, self.Uid,self.R1,self.R1C,self.R2,self.R2C,self.G1,self.G1C,self.G2,self.G2C,self.B1,self.B1C,self.B2,self.B2C)
    class Meta:
        managed = True
        db_table = 'algorithm_rgb'


class AuthGroup(models.Model):
    id = models.IntegerField(primary_key=True)  # AutoField?
    name = models.CharField(unique=True, max_length=80)

    class Meta:
        managed = False
        db_table = 'auth_group'


class AuthGroupPermissions(models.Model):
    id = models.IntegerField(primary_key=True)  # AutoField?
    group = models.ForeignKey(AuthGroup, models.DO_NOTHING)
    permission = models.ForeignKey('AuthPermission', models.DO_NOTHING)

    class Meta:
        managed = False
        db_table = 'auth_group_permissions'
        unique_together = (('group', 'permission'),)


class AuthPermission(models.Model):
    id = models.IntegerField(primary_key=True)  # AutoField?
    content_type = models.ForeignKey('DjangoContentType', models.DO_NOTHING)
    codename = models.CharField(max_length=100)
    name = models.CharField(max_length=255)

    class Meta:
        managed = False
        db_table = 'auth_permission'
        unique_together = (('content_type', 'codename'),)


class AuthUser(models.Model):
    id = models.IntegerField(primary_key=True)  # AutoField?
    password = models.CharField(max_length=128)
    last_login = models.DateTimeField(blank=True, null=True)
    is_superuser = models.BooleanField()
    first_name = models.CharField(max_length=30)
    last_name = models.CharField(max_length=30)
    email = models.CharField(max_length=254)
    is_staff = models.BooleanField()
    is_active = models.BooleanField()
    date_joined = models.DateTimeField()
    username = models.CharField(unique=True, max_length=150)

    class Meta:
        managed = False
        db_table = 'auth_user'


class AuthUserGroups(models.Model):
    id = models.IntegerField(primary_key=True)  # AutoField?
    user = models.ForeignKey(AuthUser, models.DO_NOTHING)
    group = models.ForeignKey(AuthGroup, models.DO_NOTHING)

    class Meta:
        managed = False
        db_table = 'auth_user_groups'
        unique_together = (('user', 'group'),)


class AuthUserUserPermissions(models.Model):
    id = models.IntegerField(primary_key=True)  # AutoField?
    user = models.ForeignKey(AuthUser, models.DO_NOTHING)
    permission = models.ForeignKey(AuthPermission, models.DO_NOTHING)

    class Meta:
        managed = False
        db_table = 'auth_user_user_permissions'
        unique_together = (('user', 'permission'),)


class DjangoAdminLog(models.Model):
    id = models.IntegerField(primary_key=True)  # AutoField?
    object_id = models.TextField(blank=True, null=True)
    object_repr = models.CharField(max_length=200)
    action_flag = models.PositiveSmallIntegerField()
    change_message = models.TextField()
    content_type = models.ForeignKey('DjangoContentType', models.DO_NOTHING, blank=True, null=True)
    user = models.ForeignKey(AuthUser, models.DO_NOTHING)
    action_time = models.DateTimeField()

    class Meta:
        managed = False
        db_table = 'django_admin_log'


class DjangoContentType(models.Model):
    id = models.IntegerField(primary_key=True)  # AutoField?
    app_label = models.CharField(max_length=100)
    model = models.CharField(max_length=100)

    class Meta:
        managed = False
        db_table = 'django_content_type'
        unique_together = (('app_label', 'model'),)


class DjangoMigrations(models.Model):
    id = models.IntegerField(primary_key=True)  # AutoField?
    app = models.CharField(max_length=255)
    name = models.CharField(max_length=255)
    applied = models.DateTimeField()

    class Meta:
        managed = False
        db_table = 'django_migrations'


class DjangoSession(models.Model):
    session_key = models.CharField(primary_key=True, max_length=40)
    session_data = models.TextField()
    expire_date = models.DateTimeField()

    class Meta:
        managed = False
        db_table = 'django_session'
