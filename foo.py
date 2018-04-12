from tablehost.uart import UartCom

foo = UartCom(debug=True)
foo.write("Hello, mc FOobar!\0")
print foo.readline()
