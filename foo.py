from tablehost.uart import UartCom

foo = UartCom()
foo.write(chr(0))
foo.write(chr(20))

for x in range(0,20):
	foo.write(chr(x))

bar = ''
while bar == '':
	foo.write(chr(20))
	bar = foo.read(1)
	print "Rceived: \"%s\"" % bar
