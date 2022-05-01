from django.contrib import admin
from apps.table.models import Table
from apps.table.models import Led
from apps.table.models import ColorFilter

admin.site.register(Table)
admin.site.register(Led)
admin.site.register(ColorFilter)

# Register your models here.
