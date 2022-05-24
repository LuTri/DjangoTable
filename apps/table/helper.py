from apps.table.models import ROWS, COLS


def snakish_to_coord(pos):
	if pos >= ROWS * COLS:
		raise RuntimeError("Woops. Too much Snake!")

	y, x = 0, 0
	y = int(pos / COLS)
	if y % 2 == 1:
		x = (COLS - 1) - (pos - (y * COLS))
	else:
		x = pos - (y * COLS)

	return x,y


def coord_to_snakish(x, y):
	if x > COLS or y > ROWS:
		return 255

	if y % 2 == 1:
		return (y * COLS) + (COLS - (x + 1))
	return y * COLS + x
