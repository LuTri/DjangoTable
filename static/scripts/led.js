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
	var port_height = $(document).height();

	var result_width = Math.floor(port_width / 14);
	var result_height = Math.floor(port_height / 8);

	var value = result_width < result_height ? result_width : result_height;
	value = (max_val < value ? max_val : value);

	offset_x = (port_width / 2) - ((value * 14) / 2);
	offset_y = 10;

	$('.leddiv').each(function() {
		var x, y;
		$(this).css('width', value - 2);
		$(this).css('height', value - 2);
		x = parseInt($(this).children(".pos_x").html());
		y = parseInt($(this).children(".pos_y").html());
		$(this).css('top', offset_y + y * value);
		$(this).css('left', offset_x + x * value);
	});
}

function finish_spectrum(element) {
	$(element).spectrum('destroy');
	$(element).removeClass('colorizing');
}

function colorize(element) {
};

function rgbToString(r, g, b) {
  return (
    (0x100 | Math.round(r)).toString(16).substr(1) +
    (0x100 | Math.round(g)).toString(16).substr(1) +
    (0x100 | Math.round(b)).toString(16).substr(1)
  );
};

$(document).ready(function() {
  let updateTimeout = null;
	resize();

	$(".leddiv > form > [name=color]").change(function(event) {
		var url = $(this).parent('form').attr('action');
		var color = this.jscolor.toString();

		$.ajax({
			beforeSend: function(request) {
				request.setRequestHeader('X-CSRFToken', getCookie('csrftoken'));
			},
			type: "POST",
			url: url,
			data: {'color': color}
		});


	});

	$(".leddiv").hover(function(event) {
	  $(".leddiv").children('form').children('input.jscolor').each(function (elem) {
	    this.jscolor.fromRGB(0,0,0);
	    $(this).attr('value', this.jscolor.toString());
	  });

	  let x = parseInt($(this).children('.pos_x').html());
	  let y = parseInt($(this).children('.pos_y').html());

	  let selector_x = new Array();
	  let selector_y = new Array();
	  let max_distance = Math.sqrt(
      Math.pow(2, 2) +
      Math.pow(2, 2)
    );
	  for (let s_x of [x - 2, x - 1, x, x + 1, x + 2]) {
	    selector_x.push(".pos_x.x_" + (s_x));
	  }
	  for (let s_y of [y - 2, y - 1, y, y + 1, y + 2]) {
	    selector_y.push(".pos_y.y_" + (s_y));
	  }

	  $(".leddiv").has(selector_x.join(', ')).has(selector_y.join(', ')).each(function (elem) {
      let other_x = parseInt($(this).children('.pos_x').html());
	    let other_y = parseInt($(this).children('.pos_y').html());

	    let distance = Math.sqrt(
	      Math.pow(Math.abs(other_x - x), 2) +
	      Math.pow(Math.abs(other_y - y), 2)
	    );

	    $(this).children('form').children('input.jscolor').each(function (elem) {
	      let hue = (1 / max_distance) * distance;
	      let h = (hue * 40) + 26;
	      let v = ((1 - hue) * .75) * 100;
  	    this.jscolor.fromHSV(h, 100, v);
  	    console.log('Distance: ' + distance + '; hsv: ' + h + ',1,' + v + '; result: ' + this.jscolor.toString());
        $(this).attr('value', this.jscolor.toString());
	    });
	  });

    let data = {};

    $(".jscolor").each(function(elem) {
      let led_id = $(this).parent('form').children("[name=led_id]").attr('value');
      let color = this.jscolor.toString();
      data[led_id] = color;
    });

    let url = $(this).parent().children('form').attr('action');

    if (updateTimeout != null) {
      window.clearTimeout(updateTimeout);
      updateTimeout = null;
    }

    updateTimeout = window.setTimeout(() => {
      $.ajax({
        beforeSend: function(request) {
          request.setRequestHeader('X-CSRFToken', getCookie('csrftoken'));
        },
        type: "POST",
        url: url,
        data: data
      });
    }, 50);
	});
});

$(window).resize(function() {
	resize()
});
