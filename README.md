## LED Table master

# Trivia

This Djano-application controls the [AVRMusicTable](https://github.com/LuTri/AVRMusicTable) on an Odroid U3 with an UART interface.

# Requirements

 * Python 2.7
 * A Django-compatible DB (e.g. Mysql, Postgresql)

# Installation

 * Create a user and a database in the DBMS of your choice
 * `pip install -r requirements.txt`
 * Migrate the application `python manage.py migrate`
