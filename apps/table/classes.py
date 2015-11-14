from apps.table.models import ROWS,COLS
from apps.table.models import Table
from apps.table.models import LedPos
from apps.table.models import Color
from apps.table.models import LED_CHOICES

from apps.table.helper import snakish_to_coord

from random import randint

from libs.jinja import render_to_string

class Led(object):
	def __init__(self, x, y, id = None, rgb=None):
		self.id = id
		self.x = x
		self.y = y

		try:
			self.update()
		except:
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

		return hexcol.replace(' ','0')

	def __unicode__(self):
		context = {}
		context['x'] = self.x
		context['y'] = self.y
		context['id'] = self.id
		context['color'] = self.get_rgb_hex()
		return render_to_string('table/led.html',context)

	def update(self):
		table = Table.objects.get(description = "ACTIVE")
		color = table.ledpos_set.get(pos = self.id).color
		self.rgb = {'r':int(color.r), 'g': int(color.g), 'b': int(color.b)}

class AllLeds(object):
	def __init__(self):
		self._leds = [Led(*v.split('_'),id=i) for i,v in LED_CHOICES]

	def update(self):
		for item in self._leds:
			try:
				item.update()
			except:
				pass

	def __iter__(self):
		for led in self._leds:
			yield led

LEDS = AllLeds()
