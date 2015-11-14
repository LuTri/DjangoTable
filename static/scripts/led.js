function getCookie(name) {
    var cookieValue = null;
    if (document.cookie && document.cookie != '') {
        var cookies = document.cookie.split(';');
        for (var i = 0; i < cookies.length; i++) {
            var cookie = jQuery.trim(cookies[i]);
            // Does this cookie string begin with the name we want?
            if (cookie.substring(0, name.length + 1) == (name + '=')) {
                cookieValue = decodeURIComponent(cookie.substring(name.length + 1));
                break;
            }
        }
    }
    return cookieValue;
}

function resize() {
	var max_val = 60;

	var offset_x, offset_y;

	var port_width = $(document).width();
	var port_height = $(document).height();

	var result_width = Math.floor(port_width / 14);
	var result_height = Math.floor(port_height / 8);

	var value = result_width < result_height ? result_width : result_height;
	value = (max_val < value ? max_val : value);

	offset_x = (port_width / 2) - ((value * 14) / 2);
	offset_y = (port_height / 2) - ((value * 8) / 2);

	$('.leddiv').each(function() {
		var x, y;
		$(this).css('width', value - 2);
		$(this).css('height', value - 2);
		x = parseInt($(this).children("#pos_x").html());
		y = parseInt($(this).children("#pos_y").html());
		$(this).css('top', offset_y + y * value);
		$(this).css('left', offset_x + x * value);
	});
}

function colorize(element) {
	active_led_selector = $(element).attr('id');
};

$(document).ready(function() {
	resize();
	$('.leddiv').each(function() {
		$(this).click(function() {
			colorize(this);
		});
	});
});

$(window).resize(function() {
	resize()
});
