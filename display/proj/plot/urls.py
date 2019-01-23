# -*- coding: utf-8 -*-


from django.conf.urls import url
import views
urlpatterns = [
    url(r'^$', views.pp),
    url(r'^index/$', views.index),
    url(r'^index2/$', views.index2),
    url(r'^index3/$', views.index3),
    url(r'^index4/$', views.index4),
    url(r'^index5/$', views.index5),
    url(r'^add/$', views.add),
    url('^getBigData/$', views.getBigData),
    url('^getRatioData/$', views.getRatioData),
    url('^formulaTJ/$', views.formulaTJ),
]
