from django.conf import settings
from django.http import HttpResponse, HttpResponseServerError
from django.template import RequestContext
from django.views.decorators.csrf import csrf_exempt
import json

from apps.player import CustMPDClient

def as_json(function):
	def outer(request, *args, **keywords):
		data = function(request, *args, **keywords)
		# Pass any HttpResponses (server errors?) straight through.
		if isinstance(data, HttpResponse):
			return data

		return HttpResponse(json.dumps(data, sort_keys=True, indent=4))

	return outer

@as_json
def status(request):
	with CustMPDClient.GetClient() as mpd:
		data = mpd.status()
		data.update(mpd.currentsong())
		data.update(mpd.stats())

	return data


# Playback Controls
def play(request):
	with CustMPDClient.GetClient() as mpd:
		if mpd.status().get('state', None) == 'play':
			mpd.pause()
		else:
			mpd.play()
	return HttpResponse("OK")


def prev(request):
	with CustMPDClient.GetClient() as mpd:
		mpd.previous()
	return HttpResponse("OK")



def next(request):
	with CustMPDClient.GetClient() as mpd:
		mpd.next()
	return HttpResponse("OK")



def stop(request):
	with CustMPDClient.GetClient() as mpd:
		mpd.stop()
	return HttpResponse("OK")



def play_song(request, song_id):
	with CustMPDClient.GetClient() as mpd:
		mpd.playid(song_id)
	return HttpResponse("Playing Song %s." % song_id)


# General Controls

def repeat(request):
	with CustMPDClient.GetClient() as mpd:
		repeat = int(mpd.status()['repeat'])
		mpd.repeat(0 if repeat == 1 else 1)
	return HttpResponse("Repeat: %s" % repeat)



def random(request):
	with CustMPDClient.GetClient() as mpd:
		random = int(mpd.status()['random'])
		mpd.random(0 if random == 1 else 1)

	return HttpResponse("Random: %s" % random)



def volume(request, volume):
	with CustMPDClient.GetClient() as mpd:
		mpd.setvol(volume)
		volume = mpd.status()['volume']
	return HttpResponse("Volume: %s" % volume)


# Playlist Controls
@csrf_exempt

def add_songs(request):
	post = request.POST
	with CustMPDClient.GetClient() as mpd:
		for song in json.loads(post['songs']):
			mpd.add(song)
	return HttpResponse("OK")


@csrf_exempt

def remove_songs(request):
	post = request.POST
	with CustMPDClient.GetClient() as mpd:
		for song in json.loads(post['songs']):
			song = int(song)
			mpd.deleteid(song)
	return HttpResponse("OK")


@csrf_exempt

def clear_songs(request):
	with CustMPDClient.GetClient() as mpd:
		mpd.clear()
	return HttpResponse("OK")


@csrf_exempt

def save_playlist(request):
	name = request.POST['name']
	with CustMPDClient.GetClient() as mpd:
		try:
			mpd.rm(name)
		except:
			pass
		mpd.save(name)
	return HttpResponse("OK")


# Misc Commands

def update_library(request):
	with CustMPDClient.GetClient() as mpd:
		mpd.update()
	return HttpResponse("OK")


@as_json
def playlist_info(request):
	with CustMPDClient.GetClient() as mpd:
		data={}
		data['info'] = mpd.playlistinfo()
		data['playlist'] = mpd.playlist()

	return data

@as_json
def all_songs(request, page=0):
	with CustMPDClient.GetClient() as mpd:
		mpd.iterate = True
		data = [x for x in mpd.listallinfo('/')[10*page:10*(page+1) - 1]]

	return data
