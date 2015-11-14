from django.db import models
from apps.table.helper import COLS
from apps.table.helper import ROWS
from apps.table.helper import snakish_to_coord



LED_CHOICES = [(x, u'%d_%d' % snakish_to_coord(x))for x in range(0,ROWS * COLS)]

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
		choices=LED_CHOICES,
		max_length=5
	)
	table = models.ForeignKey(Table)
	color = models.ForeignKey(Color)

	class Meta:
		unique_together = (('table', 'pos',))
# Create your models here.
