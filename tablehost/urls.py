from django.conf.urls import patterns, include, url
from django.contrib import admin

urlpatterns = patterns('',
    # Examples:
    # url(r'^$', 'tablehost.views.home', name='home'),
    # url(r'^blog/', include('blog.urls')),

    url(r'^player/', include('apps.player.urls')),
    url(r'^admin/', include(admin.site.urls)),
    url(r'^$', include('apps.main.urls')),
)
