from django.urls import include, path
from django.contrib import admin

urlpatterns =[
    path(r'player/', include('apps.player.urls')),
    path('admin/', admin.site.urls),
    path('table/', include('apps.table.urls')),
    path('', include('apps.main.urls')),
]
