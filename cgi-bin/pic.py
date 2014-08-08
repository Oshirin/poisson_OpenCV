#!/usr/local/bin/python
# -*- coding: utf-8 -*-

import cgi
import os, sys
import cv2
import time
import codecs
 
print "Content-Type: image/jpeg"

print "<html><body>"
form = cgi.FieldStorage()
text = form["path"].value

f = open(text, "rb")
b = f.read()
f.close()
print "Content-type: image/jpeg"
print 
print b

print "</html></body>"