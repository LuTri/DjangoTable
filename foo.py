from tablehost.uart import UartCom
from random import randint

foo = UartCom()
bar = [None]*(336)
while True:
	for idx in range(336):
		bar[idx] = randint(0,255)

	foo.prepare_data(bar)
	foo.write_whole_array(length=336)

