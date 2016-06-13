from django.views.decorators.csrf import ensure_csrf_cookie
from django.http import HttpResponse

from libs.jinja import render_to_response

from apps.table.models import COLS
from apps.table.models import Table
from apps.table.models import Led
from apps.table.helper import snakish_to_coord

from tablehost.uart import UartCom

import subprocess
import json

@ensure_csrf_cookie
def index(request, tableid):
	context = {}
	context['leds'] = Table.objects.get(pk=tableid).get_leds()
	context['max_cols'] = COLS
	return render_to_response('table.html', context)

def setcol(request, ledid, tableid):
	mccom = UartCom(True)
	color = request.POST['color']

	table, created = Table.objects.get_or_create(pk=tableid)

	led, created = Led.objects.get_or_create(table=table, pos=int(ledid))
	led.color = color

	led.save()

	mccom.prepare_data(table.to_uart_array())
	mccom.write_whole_array()

	return HttpResponse('')

# Create your views here.
