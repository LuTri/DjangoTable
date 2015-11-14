ROWS = 8
COLS = 14

def snakish_to_coord(pos):
	if pos >= ROWS * COLS:
		raise RuntimeError("Woops. Too much Snake!")

	y,x = 0,0
	y = pos / COLS
	if (y % 2 == 1):
		x = (COLS - 1) - (pos - (y * COLS))
	else:
		x = pos - (y * COLS)

	return x,y
