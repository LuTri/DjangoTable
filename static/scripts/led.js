var url = '/table/setcol/';
var cont_width = 195;
var cont_height = 274;

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

	var value = Math.floor(port_width / 14);

	value = (max_val < value ? max_val : value);

	offset_x = (port_width / 2) - ((value * 14) / 2);

	$('.leddiv').each(function() {
		var x, y;
		$(this).css('width', value - 2);
		$(this).css('height', value - 2);
		x = parseInt($(this).children("#pos_x").html());
		y = parseInt($(this).children("#pos_y").html());
		$(this).css('top', y * value);
		$(this).css('left', offset_x + x * value);
	});
}

function finish_spectrum(element) {
	$(element).spectrum('disable');
	$(element).spectrum('hide');
	$(element).spectrum('destroy');
	$(element).removeClass('colorizing');
}

function colorize(element) {
	$('.leddiv > .color_sel.colorizing').each(function() {
		finish_spectrum(this);
	});

	var selector = $(element).children('.color_sel');

	selector.addClass('colorizing');
	selector.spectrum({
		flat: true,
		showPalette: true,
		preferredFormat: "hex",
		clickoutFiresChange: false,
		allowEmpty: true,
		previewDiv: element,
		color: $(element).css('background-color'),
		change: function() {
			var result = selector.spectrum('get');
			var posturl = url + selector.parent().attr('id').substr(3) + '/';
			finish_spectrum(selector);

			if (!(result === null || result === undefined)) {
				var color = result.toHex();
				$.ajax({
					beforeSend: function(request) {
						request.setRequestHeader('X-CSRFToken', getCookie('csrftoken'));
					},
					type: "POST",
					url: posturl,
					data: {'color': color}
				});
			}
		}
	});
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
