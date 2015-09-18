import mpd

class CustMPDClient(mpd.MPDClient):
	@staticmethod
	def GetClient():
		newclient = CustMPDClient(use_unicode=True)
		newclient.connect("localhost",6600)

		return newclient

	def __enter__(self):
		return self

	def __exit__(self, exc_type, exc_val, exc_tb):
		self.disconnect()
