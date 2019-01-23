from __future__ import unicode_literals

from django.db import models
import datetime
import django.utils.timezone as timezone
# Create your models here.
 
 
class RGB(models.Model):
    Uid = models.CharField(max_length=30,default='123456789')
    Timestamp = models.IntegerField(default=1000000000)
    Datetime = models.DateTimeField(default = timezone.now)
    R1 = models.IntegerField(default=-999)
    R1C= models.IntegerField(default=-999)  
    R2 = models.IntegerField(default=-999)
    R2C = models.IntegerField(default=-999)  
    G1 = models.IntegerField(default=-999)
    G1C = models.IntegerField(default=-999)  
    G2 = models.IntegerField(default=-999)
    G2C = models.IntegerField(default=-999)  
    B1 = models.IntegerField(default=-999)
    B1C = models.IntegerField(default=-999)  
    B2 = models.IntegerField(default=-999)
    B2C = models.IntegerField(default=-999)  
    Statue1=models.IntegerField(default=10)
    Statue2=models.IntegerField(default=10)
    Statue3=models.IntegerField(default=10)

    def __unicode__(self):
        
        return u'%s--- %s---%d-%d-%d-%d-%d-%d-%d-%d-%d-%d-%d-%d' % (self.Datetime, self.Uid,self.R1,self.R1C,self.R2,self.R2C,self.G1,self.G1C,self.G2,self.G2C,self.B1,self.B1C,self.B2,self.B2C)

    