#!/bin/bash

git submodule init
git submodule update
git submodule foreach --recursive git fetch origin
git submodule foreach --recursive git submodule init
git submodule foreach --recursive git submodule update
