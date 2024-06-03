#!/bin/bash


# for therminal the first 2 numbers are in charaters (i.e. 80x20 is 80
# characters wide and 20 haracters high
# the next 2 numbers are distance from left and distance from top in pixels
# use move a window to where you want it to go and use xwininfo to find its location in pixels

xfce4-terminal -T queue0 --geometry 80x20+0+1
xfce4-terminal -T queue1 --geometry 80x20+820+1


