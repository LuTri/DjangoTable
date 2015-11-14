from django.db import models

ROWS = 8
COLS = 14

LED_CHOICES = ['%d_%d' % (x,y) for x in range(0,COLS) for y in range(0,ROWS)]

class Table(models.Model):	
	description = models.CharField(
		max_length=200,
		verbose_name='ColorScheme'
	)

class Color(models.Model):
	descr = models.CharField(
		max_length=200,
		verbose_name='Color "Name"',
		null=False,
		default='',
	)
	r = models.IntegerField(default=0)
	g = models.IntegerField(default=0)
	b = models.IntegerField(default=0)

	def __unicode__(self):
		return self.descr

	class Meta:
		unique_together = (('r','g','b',))

class LedPos(models.Model):
	pos = models.CharField(
		choices=enumerate(LED_CHOICES),
		max_length=5
	)
	table = models.ForeignKey(Table)
	color = models.ForeignKey(Color)
# Create your models here.
