from django.shortcuts import render
from libs.jinja import render_to_response

import subprocess
import json

def HsvToRgb(h,s,v):
	r,g,b = None,None,None

	while (h < 0):
		h += 360.0

	while (h > 360):
		h -= 360.0

	if (v <= 0):
		r = g = b = 0

	elif (s <= 0):
		r = g = b = v;
	else:
		hf = h / 60.0
		hf_i = int(hf)
		f = hf - hf_i

		pv = v * (1 - s)
		qv = v * (1 - s * f)
		tv = v * (1 - s * (1-f))

		if hf_i == 0:
			r = v
			g = tv
			b = pv
		elif hf_i == 1:
			r = qv
			g = v
			b = tv
		elif hf_i == 2:
			r = pv
			g = v
			b = tv
		elif hf_i == 3:
			r = pv
			g = qv
			b = v
		elif hf_i == 4:
			r = tv
			g = pv
			b = v
		elif hf_i == 5:
			r = v
			g = pv
			b = qv
		else:
			r = g = b = v

	return int(r*255),int(g*255),int(b*255)

def index(request):
	foo = json.load(open('tmp','r'))
	vals = []
	bar = json.loads(subprocess.check_output(['/home/tristan/projects/ledtable/mc/testing/test','%d' % (foo['val'] * 3)]))
	for row in range(0,8):
		for col in range(0,14):
			vals.append((bar['values']['%d' % col]['%d' % row]['r'], bar['values']['%d' % col]['%d' % row]['g'], bar['values']['%d' % col]['%d' % row]['b'],))

	foo['val'] = foo['val'] + 1
	if (foo['val'] > 300):
		foo['val'] = 0
	json.dump(foo, open('tmp','w'))
	return render_to_response('table_test.html',{'vals':vals})
	

# Create your views here.
