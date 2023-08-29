==========================
Frequently Asked Questions
==========================

Why is my code giving errors?
-----------------------------

Pycollo's error checking for bounds and guesses can give opaque error messages or even fail. Make sure that you have provided bounds and a guess for every variable and constraint where necessary. It can be helpful to look at the dimensions of all attributes and compare this to the bounds and guesses that you have supplied to ensure that these are equal. If they are not then you are probably missing some bounds or a guess.
