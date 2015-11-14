from django.conf.urls import patterns, include, url
from django.contrib import admin

urlpatterns = patterns('',
    url(r'^player/', include('apps.player.urls')),
    url(r'^admin/', include(admin.site.urls)),
    url(r'^table/', include('apps.table.urls')),
    url(r'^$', include('apps.main.urls')),
)
