from django.urls import path
from apps.player import ajax

urlpatterns = [
    path('update_library/', ajax.update_library),
    path('save_playlist/', ajax.save_playlist),
    path('clear_songs/', ajax.clear_songs),
    path('remove_songs/', ajax.remove_songs),
    path('add_songs/', ajax.add_songs),
    path('volume/<int:volume>/', ajax.volume),
    path('random/', ajax.random),
    path('repeat/', ajax.repeat),
    path('play_song/<int:song_id>/', ajax.play_song),
    path('stop/', ajax.stop),
    path('prev/', ajax.prev),
    path('play/', ajax.play),
    path('status/', ajax.status),
    path('update_library/', ajax.update_library),
    path('playlist_info/', ajax.playlist_info),
    path('all_songs/', ajax.all_songs),
]
