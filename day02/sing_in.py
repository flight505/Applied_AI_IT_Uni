#!/usr/bin/env python3

##########################################################
# Copyright (c) Jesper Vang <jesper_vang@me.com>         #
# Created on 3 Aug 2021                                 #
# Version:	0.0.1                                        #
# What: ? 						                         #
##########################################################

import os

os.system("cls||clear")  # this line clears the screen 'cls' = windows 'clear' = unix
import sys

# from pprint import pprint - use pprint() to pretty print

import re
from mechanize import Browser


import urllib

#!/usr/bin/python
import re
from mechanize import Browser

br = Browser()

# Ignore robots.txt
br.set_handle_robots(False)
# Google demands a user-agent that isn't a robot
br.addheaders = [("User-agent", "Firefox")]

# Retrieve the Google home page, saving the response
br.open("https://webinar.tapmeetinglive.com/events/Ming_Wei_Chang/join?ref=index")

# Select the search box and search for 'foo'
br.select_form("Email")
br.form["q"] = "jesper_vang@me.com"

# Get the search results
br.submit()

# Find the link to foofighters.com; why did we run a search?
# resp = None
# for link in br.links():
#     siteMatch = re.compile( 'www.foofighters.com' ).search( link.url )
#     if siteMatch:
#         resp = br.follow_link( link )
#         break

# Print the site
content = resp.get_data()
print(content)
