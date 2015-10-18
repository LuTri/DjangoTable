from tablehost.uart import UartCom
from random import randint

foo = UartCom(debug=True)
bar = [None]*(42)
while True:
	for idx in range(42):
		bar[idx] = randint(0,50)

	foo.prepare_data(bar)
	foo.write_whole_array(length=42)

