# README - PDF Keyword searching tool #

This README will help get you set up and running with this tool.

### What does it do? ###

This tool searches a folder of PDF documents with a given document containing keywords.
It produces an output where each document is ranked according to how many hits on the keywords it has, with highest 
order going to words that match the expected usage (noun, verb etc), followed by word matches and 
finally stemmed words.

The main report will also show the stemmed words that were matched and the exact matches in the context of
the sentence they were in within each document.

A second report shows a list of people and organisations identified within the top 10% of scored documents.

### How do I get set up? ###

You will need:
A tika server running (for running offline)
en_core_web_sm loaded for Spacy (for running offline)
The DejaVU sans font suite installed 

Usage is from the command line and takes the following arguments:
-i "C:\input" -- the folder of PDF files to trawl through
-o "C:\outpout"  -- the folder where the reports should be saved
-k "C:\keywords.txt"  -- the location of the keywords list (each keyword (or set of words to match, up to 3) should be on a new line)

E.G python processtext.py -i "C:\input" -o "C:\output"  -k "C:\keywords.txt"

### Who do I talk to? ###

TIDE Software team