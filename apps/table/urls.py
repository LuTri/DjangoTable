from django.conf.urls import patterns, include, url
from django.contrib import admin

from apps.table.views import *

urlpatterns = patterns('',
    url(r'^setcol/(?P<ledid>\d+)/$', setcol, name='setcol'),
    url(r'^$', index, name='table_main'),
)
