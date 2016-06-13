from django.views.decorators.csrf import ensure_csrf_cookie
from django.http import HttpResponse

from libs.jinja import render_to_response

from apps.table.classes import LEDS
from apps.table.models import COLS
from apps.table.models import Table
from apps.table.models import Color
from apps.table.models import LedPos
from apps.table.models import LED_CHOICES
from apps.table.helper import snakish_to_coord

from tablehost.uart import UartCom

import subprocess
import json

@ensure_csrf_cookie
def index(request):
	context = {}
	context['leds'] = LEDS
	context['max_cols'] = COLS
	return render_to_response('table.html', context)

def setcol(request, ledid):
	mccom = UartCom(False)
	hex_ = request.POST['color']
	r = int(hex_[0:2], 16)
	g = int(hex_[2:4], 16)
	b = int(hex_[4:6], 16)

	table, created = Table.objects.get_or_create(description="ACTIVE")
	color, created = Color.objects.get_or_create(r=r,g=g,b=b)

	x,y = snakish_to_coord(int(ledid))
	pos = [i for i, v in enumerate(LED_CHOICES) if v == '%d_%d' % (x,y)][0]

	led = LedPos.objects.get_or_create(table=table, color=color, pos=pos)

	print (r,g,b,)

	mccom.prepare_data(table.to_uart_array())
	mccom.write_whole_array()

	return HttpResponse('')

# Create your views here.
