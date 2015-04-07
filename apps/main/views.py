from django.shortcuts import render
from libs.jinja import render_to_response

def index(request):
	return render_to_response('index.html')
	

# Create your views here.
