var EventServiceO = new EventService_cl();

var all_images = {
	'back': [
		'/static/images/player/player_back.png',
		'/static/images/player/player_back_doing.png',
		'/static/images/player/player_back_possible.png',
		'/static/images/player/player_back_impossible.png',
	],
	'forward': [
		'/static/images/player/player_forward.png',
		'/static/images/player/player_forward_doing.png',
		'/static/images/player/player_forward_possible.png',
		'/static/images/player/player_forward_impossible.png',
	],
	'pause': [
		'/static/images/player/player_pause.png',
		'/static/images/player/player_pause_doing.png',
		'/static/images/player/player_pause_possible.png',
		'/static/images/player/player_pause_impossible.png',
	],
	'play': [
		'/static/images/player/player_play.png',
		'/static/images/player/player_play_doing.png',
		'/static/images/player/player_play_possible.png',
		'/static/images/player/player_play_impossible.png',
	],
	'stop': [
		'/static/images/player/player_stop.png',
		'/static/images/player/player_stop_doing.png',
		'/static/images/player/player_stop_possible.png',
		'/static/images/player/player_stop_impossible.png'
	]
};


var Player_cl = Class.create({
	initialize : function () {
		this.playing = false;
		this.paused = false;
		this.list_pos = 0;
		this.shuffle = 0;
		this.repeat = 0;

		this.songs = [];
	},
	register_elements : function(btn_play, btn_stop, btn_prev, btn_forward,
		list_songs, list_lists, disp_title, disp_artist, time_passed,
		time_act, btn_shuffle, btn_repeat) {

		this.player_btns = [btn_play, btn_stop, btn_prev, btn_forward];
		this.state_btns = [btn_shuffle, btn_repeat];
		
		this.disp_divs = [disp_title, disp_artist];
		this.time_spans = [time_passed, time_act];
	},
	fetch_songs : function() {
		
	},
	fetch_state : function) {
	}
});
