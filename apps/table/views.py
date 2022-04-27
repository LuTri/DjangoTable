import socket

from django.views.decorators.csrf import ensure_csrf_cookie
from django.views.decorators.http import require_POST

from django.http import HttpResponse

from libs.jinja import render_to_response

from apps.table.models import COLS
from apps.table.models import Table
from apps.table.models import Led
from apps.table.helper import snakish_to_coord

from tablehost.uart import UartSimulator

import subprocess
import json

@ensure_csrf_cookie
def index(request, tableid):
	context = {}
	table = Table.objects.get(pk=tableid)
	context['table'] = table
	context['leds'] = table.get_leds()
	context['max_cols'] = COLS
	return render_to_response('table.html', context)

@require_POST
def update_table(request, tableid):
	table = Table.objects.get(pk=tableid)

	for led in table.led_set.all():
		led.color = request.POST.get(f'{led.pos}', '000000')
		led.save()

	try:
		mccom = UartSimulator(True)
		mccom.prepare_data(table.to_uart_array())
		mccom.write_whole_array()
	except socket.error as exc:
		print(f'{exc}')

	return HttpResponse(status=200)

def setcol(request, ledid, tableid):
	color = request.POST['color']
	do_push = request.POST.get('push', False)

	table, created = Table.objects.get_or_create(pk=tableid)

	led, created = Led.objects.get_or_create(table=table, pos=int(ledid))
	led.color = color

	led.save()

	if do_push:
		try:
			mccom = UartSimulator(True)
			mccom.prepare_data(table.to_uart_array())
			mccom.write_whole_array()
		except socket.error as exc:
			print(f'{exc}')

	return HttpResponse('')

# Create your views here.
