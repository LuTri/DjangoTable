# -*- coding: utf-8 -*-

from django.http import HttpResponse
from django.conf import settings
from django.core.urlresolvers import get_callable
from os.path import join
from jinja2.loaders import FileSystemLoader, PackageLoader, ChoiceLoader
import re
from jinja2 import Environment

__all__ = ('render_to_string', 'render_to_response', 'render_tex_to_string')

loader_array = []
for pth in getattr(settings, 'TEMPLATE_DIRS', ()):
	loader_array.append(FileSystemLoader(join(settings.BASE_DIR, pth)))

default_mimetype = getattr(settings, 'DEFAULT_CONTENT_TYPE')
global_exts = getattr(settings, 'JINJA_EXTS', ())

LATEX_SUBS = (
	(re.compile(r'\\'), r'\\textbackslash'),
	(re.compile(r'([{}_#%&$])'), r'\\\1'),
	(re.compile(r'~'), r'\~{}'),
	(re.compile(r'\^'), r'\^{}'),
	(re.compile(r'"'), r"''"),
	(re.compile(r'\.\.\.+'), r'\\ldots'),
)

def escape_tex(value):
	if isinstance(value,basestring):
		newval = value
	else:
		newval = unicode(value)

	for pattern, replacement in LATEX_SUBS:
		newval = pattern.sub(replacement, newval)
	return newval

class LazyEnv(object):
	def __init__(self):
		self._environment = None

	def __getattr__(self,name):
		if self._environment == None:
			self._environment = Environment(extensions=global_exts, loader=ChoiceLoader(loader_array))
			self._environment.globals['STATIC_URL'] = settings.STATIC_URL
			self._environment.globals['url_reverse'] = get_callable('django.core.urlresolvers.reverse_lazy')
		if name == '__members__':
			return dir(self._environment)
		return getattr(self._environment, name)

class LazyTexEnv(object):
	def __init__(self):
		self._environment = None

	def __getattr__(self,name):
		if self._environment == None:
			self._environment = Environment(extensions=global_exts, loader=ChoiceLoader(loader_array))
			self._environment.block_start_string = '((*'
			self._environment.block_end_string = '*))'
			self._environment.variable_start_string = '((('
			self._environment.variable_end_string = ')))'
			self._environment.comment_start_string = '((='
			self._environment.comment_end_string = '=))'
			self._environment.filters['escape_tex'] = escape_tex
		if name == '__members__':
			return dir(self._environment)
		return getattr(self._environment, name)

texenv = LazyTexEnv()
env = LazyEnv()

def render_to_string(filename, context={}):
	template = env.get_template(filename)
	rendered = template.render(**context)
	return rendered

def render_to_response(filename, context={}, request=None, mimetype=default_mimetype):
	if request:
		context['request'] = request
		context['user'] = request.user
	rendered = render_to_string(filename, context)
	return HttpResponse(rendered,content_type=mimetype)

def render_tex_to_string(filename, context={}, request=None, mimetype=default_mimetype):
	template = texenv.get_template(filename)
	rendered = template.render(**context)
	return rendered
