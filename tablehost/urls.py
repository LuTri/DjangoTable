from django.conf.urls import include, url
from django.contrib import admin
from apps.table.views import index

urlpatterns =[
    url(r'^player/', include('apps.player.urls')),
    url(r'^admin/', include(admin.site.urls)),
    url(r'^test$', index, name='test'),
    url(r'^$', include('apps.main.urls')),
)
