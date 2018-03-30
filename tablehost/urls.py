from django.conf.urls import include, url
from django.contrib import admin

urlpatterns =[
    url(r'^player/', include('apps.player.urls')),
    url(r'^admin/', include(admin.site.urls)),
    url(r'^', include('apps.main.urls')),
]
