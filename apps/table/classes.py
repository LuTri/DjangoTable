from apps.table.models import ROWS,COLS
from apps.table.helper import snakish_to_coord

from random import randint

from libs.jinja import render_to_string

class Led():
	def __init__(self, x, y, id = None, rgb=None):
		self.id = id
		self.x = x
		self.y = y
		self.rgb = rgb or {
			'r':randint(0,255),
			'g':randint(0,255),
			'b': randint(0,255)
		}

	def set_rgb(self, rgb):
		self.rgb = rgb

	def get_rgb_hex(self):
		hexcol = ""
		r = hex(self.rgb['r'])
		g = hex(self.rgb['g'])
		b = hex(self.rgb['b'])

		hexcol += '%02s' % r.split('x')[1]
		hexcol += '%02s' % g.split('x')[1]
		hexcol += '%02s' % b.split('x')[1]

		hexcol = hexcol.replace(' ','0')
		return hexcol

	def __unicode__(self):
		context = {}
		context['x'] = self.x
		context['y'] = self.y
		context['id'] = self.id
		context['color'] = self.get_rgb_hex()
		return render_to_string('table/led.html',context)
		

LEDS = [Led(*snakish_to_coord(x),id=x) for x in range(0, ROWS * COLS)]
