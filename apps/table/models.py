from django.db import models
from libs.jinja import render_to_string

ROWS = 8
COLS = 14

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
			result[pos * 3 + 1], result[pos * 3], result[pos * 3 + 2] =\
				self.hex_to_rgb(item.color)
		return result

	def get_leds(self):
		result = [Led.objects.get_or_create(pos=i, table=self)[0]\
			for i in range(0,COLS*ROWS)]
		return result
			

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
