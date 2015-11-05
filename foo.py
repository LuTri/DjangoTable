from tablehost.uart import UartCom
from random import randint

foo = UartCom(debug=True)
bar = [None]*(336)

def fill_arr(length,val):
	bar = [None]*length
	for i in range(0,length):
		bar[i] = val
	return bar

array = fill_arr(336,1)

foo.prepare_data(array)
