import sys
import datetime

def printCurrentTime(message):
  print(message)
  print(datetime.datetime.now())
  sys.stdout.flush()