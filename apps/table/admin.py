from django.contrib import admin
from apps.table.models import Table
from apps.table.models import Color
from apps.table.models import LedPos

admin.site.register(Table)
admin.site.register(LedPos)
admin.site.register(Color)

# Register your models here.
