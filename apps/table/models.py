import os
import math

import colorsys

from django.db import models
from libs.jinja import render_to_string

ROWS = 8
COLS = 14


class ColorFilter(models.Model):
    description = models.CharField(
        max_length=200,
        null=True,
        default=None
    )

    fnc_string = models.TextField(
        null=True,
        default=None,
    )

    applied_in = models.ManyToManyField(
        'Table',
        related_name='color_filters',
        blank=True,
    )

    def apply(self, r, g, b):
        data = {}
        _fnc = os.linesep.join(f'    {line}' for line in self.fnc_string.split(os.linesep))
        _fnc = f'def _fnc(r,g,b):{os.linesep}{_fnc}'

        exec(_fnc, data)
        return data['_fnc'](r, g, b)

    def __str__(self):
        return self.description


class Table(models.Model):	
    description = models.CharField(
        max_length=200,
        verbose_name='ColorScheme'
    )
    def hex_to_rgb(self, hex):
        r = int(hex[0:2], 16)
        g = int(hex[2:4], 16)
        b = int(hex[4:6], 16)
        return r,g,b

    def to_uart_array(self):
        result = [0] * ROWS * COLS * 3
        for item in self.led_set.all():
            pos = int(item.pos)
            r, g, b = self.hex_to_rgb(item.color)
            for _filter in self.color_filters.all():
                r, g, b = _filter.apply(r, g, b)

            result[pos * 3 + 1] = int(r)
            result[pos * 3] = int(g)
            result[pos * 3 + 2] = int(b)

        return result

    def get_leds(self):
        result = [Led.objects.get_or_create(pos=i, table=self)[0]\
            for i in range(0,COLS*ROWS)]
        return result


class FrameOrder(models.Model):
    previous_frame = models.ForeignKey(
        'AnimationFrame',
        related_name='order_before',
        null=True,
        default=None,
        on_delete=models.DO_NOTHING
    )
    this_frame = models.ForeignKey(
        'AnimationFrame',
        related_name='order',
        null=False,
        on_delete=models.CASCADE
    )
    next_frame = models.ForeignKey(
        'AnimationFrame',
        related_name='order_after',
        null=True,
        default=None,
        on_delete=models.DO_NOTHING
    )


class AnimationFrame(models.Model):
    delta_fnc = models.TextField(
        null=True,
        default=None,
        blank=True,
    )

    raw_matrix = models.JSONField(null=True, default=None, blank=True)
    table = models.ForeignKey(
        Table, null=True, default=None, on_delete=models.RESTRICT, blank=True
    )

    @classmethod
    def generate_default_matrix(cls):

        def _color(distance, max_distance):
            r, g, b = colorsys.hsv_to_rgb(1 / max_distance * distance, 1, 1)
            return r * 255, g * 255, b * 255

        max_distance = math.sqrt(ROWS**2 + COLS**2)
        for n in range(ROWS):
            colors = [_color(math.sqrt(n**2 + c**2), max_distance) for c in range(COLS)]
            yield colors

    def generate_matrix_rows(self):
#        if self.delta_fnc is None and self.raw_matrix is None and self.table is None:

        for colors in self.generate_default_matrix():
            yield colors


class Led(models.Model):
    pos = models.IntegerField(default=0)
    table = models.ForeignKey(Table, on_delete=models.RESTRICT)
    color = models.CharField(max_length=6, default='000000')

    def __str__(self):
        from apps.table.helper import snakish_to_coord
        context = {}
        context['x'], context['y'] = snakish_to_coord(self.pos)
        context['id'] = self.pos
        context['color'] = self.color
        context['table'] = self.table.pk
        return render_to_string('table/led.html',context)
