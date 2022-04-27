from django.urls import path

from apps.table.views import *

urlpatterns = [
    path('update/<int:tableid>/', update_table, name='update_table'),
    path('setcol/<int:ledid>/<int:tableid>/', setcol, name='setcol'),
    path('<int:tableid>/', index, name='table_main'),
]
