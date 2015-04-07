/* prefetch images */

var all_images = {
	'back': [
		'/static/images/player/player_back.png',				//0
		'/static/images/player/player_back_doing.png',			//1
		'/static/images/player/player_back_possible.png',		//2
		'/static/images/player/player_back_impossible.png',		//3
	],
	'forward': [
		'/static/images/player/player_forward.png',				//4
		'/static/images/player/player_forward_doing.png',		//5
		'/static/images/player/player_forward_possible.png',	//6
		'/static/images/player/player_forward_impossible.png',	//7
	],
	'pause': [
		'/static/images/player/player_pause.png',				//8
		'/static/images/player/player_pause_doing.png',			//9
		'/static/images/player/player_pause_possible.png',		//10
		'/static/images/player/player_pause_impossible.png',	//11
	],
	'play': [
		'/static/images/player/player_play.png',				//12
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

var indizes = {
	'blank': 0,
	'doing': 1,
	'possible': 2,
	'impossible': 3
}

var act_loading = 0;

var elements = {};

var player_state = {
	'playing': true,
	'paused': false,
	'list_pos': 0,
	'shuffle': 0,
	'repeat': 0
};

function set_button(btn_id, type, state) {
	var btn = $(btn_id + ' img');
	btn.attr('src',all_images[type][indizes[state]]);
}

function refresh_state() {
	if (player_state.playing) {
		set_button('#player_play','play','doing');
	} else {
		if (player_state.paused) {
			set_button('#player_play','pause','blank');
		} else {
			set_button('#player_play', 'play','blank');
		}
	}
}

function f_playpause () {
	if (player_state.playing) {
		player_state.paused = true;
		player_state.playing = false;
	} else {
		player_state.paused = false;
		player_state.playing = true;
	}
	
	refresh_state();
};

function f_forward () {
}

function f_backward () {
}

function f_stop () {
	player_state.paused = false;
	player_state.playing = false;

	refresh_state();
}

function hilight_possible(element) {
	if (element == elements.stop && (player_state.playing || player_state.paused)) {
		set_button('#player_stop','stop','possible');
	}
}

var Player = new Player_cl();

$(document).ready(function () {
	Player.register_elements(
		$('#player_play'),
		$('#player_stop'),
		$('#player_forward'),
		$('#player_backward'),
		$('#list_songs'),
		$('#list_list'),
		$('#disp_title'),
		$('#disp_artist'),
		$('#time_passed'),
		$('#time_act'),
		$('#player_shuffle'),
		$('#player_repeat')
	);
});
