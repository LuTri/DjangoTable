from django.conf.urls import include, url
from apps.player import ajax

urlpatterns = [
    url(r'^update_library/$', ajax.update_library),
    url(r'^save_playlist/$', ajax.save_playlist),
    url(r'^clear_songs/$', ajax.clear_songs),
    url(r'^remove_songs/$', ajax.remove_songs),
    url(r'^add_songs/$', ajax.add_songs),
    url(r'^volume/(?P<volume>\d{1,3})/$', ajax.volume),
    url(r'^random/$', ajax.random),
    url(r'^repeat/$', ajax.repeat),
    url(r'^play_song/(?P<song_id>\d+)/$', ajax.play_song),
    url(r'^stop/$', ajax.stop),
    url(r'^prev/$', ajax.prev),
    url(r'^play/$', ajax.play),
    url(r'^status/$', ajax.status),
    url(r'^update_library/$', ajax.update_library),
    url(r'^playlist_info/$', ajax.playlist_info),
    url(r'^all_songs/$', ajax.all_songs),
]
