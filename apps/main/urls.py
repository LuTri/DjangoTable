from django.conf.urls import url
from apps.main import views

urlpatterns = [
    url(r'^$', views.index, name="index"),
]
