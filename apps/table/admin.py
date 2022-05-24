from django.contrib import admin
from django.utils.html import format_html
from apps.table.models import Table
from apps.table.models import Led
from apps.table.models import ColorFilter
from apps.table.models import AnimationFrame

admin.site.register(Table)
admin.site.register(Led)
admin.site.register(ColorFilter)


@admin.register(AnimationFrame)
class AnimationFrameAdmin(admin.ModelAdmin):
    class Media:
        css = {
            'all': ('css/admin/frame_preview.css',)
        }
    _color_template = '<span class="admin_preview_pixel" style="background-color: rgb({},{},{});">&nbsp;</span>'
    _row_template = '<div class="admin_preview_row">{}</div>'
    model = AnimationFrame
    readonly_fields = ('preview',)

    def preview(self, obj):
        _html = ''
        for row in obj.generate_matrix_rows():
            _row = ''
            for color in row:
                _row = ''.join([_row, self._color_template.format(*color)])
            _html = ''.join([_html, self._row_template.format(_row)])
        return format_html(_html)