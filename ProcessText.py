import glob
from tika import parser
import os
import nltk
from nltk import word_tokenize
from nltk import sent_tokenize
from langdetect import detect
import pandas as pd
import string
import re
from nltk.corpus import stopwords
from collections import defaultdict
from nltk.stem import PorterStemmer
import en_core_web_sm  # or any other model you downloaded via spacy download or pip
from fpdf import FPDF
import sys, getopt

input_path = ''
output_path = ''
keywords_path = ''

def main(argv):
    try:
        opts, args = getopt.getopt(argv,"hi:o:k:",["ifile=","ofile=","keywords="])
        print(opts)
    except getopt.GetoptError:
        print('ProcessText.py -i <inputfile> -o <outputfile> -k <keywords>')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print('ProcessText.py -i <inputfile> -o <outputfile> -k <keywords>')
            sys.exit()
        elif opt in ("-i", "--ifile"):
            global input_path
            input_path = arg
        elif opt in ("-o", "--ofile"):
            global output_path
            output_path = arg
        elif opt in ("-k", "--keywords"):
            global keywords_path
            keywords_path = arg


main(sys.argv[1:])


# Set up PDF file - requires the DejaVu font to be installed in a fonts folder in the
# fpdf package directory in the python environment
def buildPDF():
    pdf = FPDF()
    pdf.add_page()
    pdf.add_font('DejaVu', '', 'DejaVuSans.ttf', uni=True)
    pdf.add_font('DejaVuSans-Bold', '', 'DejaVuSans-Bold.ttf', uni=True)
    pdf.add_font('DejaVuSans-Oblique', '', 'DejaVuSans-Oblique.ttf', uni=True)
    pdf.add_font('DejaVuSans-BoldOblique', '', 'DejaVuSans-BoldOblique.ttf', uni=True)
    pdf.set_font('DejaVuSans-Bold', '', 14)
    pdf.cell(w=0, txt="Output Report", ln=1, align="C")
    pdf.ln(20)
    return pdf


# Load the model to be used by SpaCy for Named Entity Recognition
nlp = en_core_web_sm.load()
# Set up a stemmer to use to stem words
pstemmer = PorterStemmer()


# Set up input path, stop words to be excluded and keywords to be matched
stop_words = set(stopwords.words('english'))


# Get keywords from file
def file_read(keywordlist):
    content_array = []
    with open(keywordlist) as f:
        # Content_list is the list that contains the read lines.
        for line in f:
            content_array.append(str.rstrip(line))
        return content_array


keywords = file_read(keywords_path)
keywords_bigrams = [w for w in keywords if len(w.split()) == 2]
keywords_bigrams = [w.lower() for w in keywords_bigrams]
keywords_trigrams = [w for w in keywords if len(w.split()) == 3]
keywords_trigrams = [w.lower() for w in keywords_trigrams]

# TODO: print these with the output to show any missing data
missed_keywords = [w for w in keywords if len(w.split()) < 1 or len(w.split()) > 3]

# Filter out stopwords from keywords list, POS tag keywords
filterkeywords = [w for w in keywords if w not in stop_words and len(w.split()) == 1]
poskeywords = nltk.pos_tag(filterkeywords)
filterkeywords = [w.lower() for w in filterkeywords]


# If the first keyword is a verb, move it and reparse the list
# This prevents verbs that may also be nouns being misidentified
if poskeywords[0][1] == 'VBZ':
    filterkeywords.insert(1, filterkeywords.pop(0))
    poskeywords = nltk.pos_tag(filterkeywords)

# Build a list of stem keywords for matching
stemkeywords = [(pstemmer.stem(t),t) for t in filterkeywords]

print(stemkeywords)

# Set up Dataframe - this will hold all the documents and the scores
d = pd.DataFrame()


word_matches = defaultdict(list)
bigram_matches = defaultdict(list)
trigram_matches = defaultdict(list)
pos_matches = defaultdict(list)


# Use Tika to parse the file
def parsewithtika(inputfile):
    parsed = parser.from_file(inputfile)
    # Extract the text content from the parsed file
    psd = parsed["content"]
    return re.sub(r'\s+', ' ', psd)


# Language filter - removes non english documents from the list
def filterlanguage(inputfile):
    if detect(inputfile) != 'en':
        return True
    return False


# Get parts of speech from SpaCy
def pos(x):
    return [(token.text, token.tag_) for token in x]


# Add the parts of speech to the words
def spacy_pos(x):
    pos_sent = []
    for sentence in x:
        processed_spacy = nlp(sentence)
        pos_sent.append(pos(processed_spacy))
    return pos_sent


# Add NER tags to words, and return the set so we don't have duplicates
def ner(x):
    ents = []
    for sentence in x:
        processed_spacy = nlp(sentence)
        for ent in processed_spacy.ents:
            ents.append((ent.text, ent.label_))
    return set(ents)


# Return bigrams for matching
def bigrams(x):
    bigram = list(nltk.ngrams(x, 2))
    return bigram

# Return trigrams for matching
def trigrams(x):
    trigram = list(nltk.ngrams(x, 3))
    return trigram

# Word tokens, parts of speech tagging
def wordtokens(dataframe):
    # Get all the words
    dataframe['words'] = (dataframe['sentences'].apply(lambda x: [word_tokenize(item) for item in x]))
    # Get all the parts of speech tags
    dataframe['pos'] = dataframe['sentences'].map(spacy_pos)
    # Get all the named entity tags
    dataframe['ner'] = dataframe['sentences'].map(ner)
    # Lowercase every word and put them all in a single list for each document
    dataframe['allwordsorig'] = dataframe['words'].apply(lambda x: [item.strip(string.punctuation).lower()
                                                                for sublist in x for item in sublist])
    # Strip out non words and stop words
    dataframe['allwords'] = (dataframe['allwordsorig'].apply(lambda x: [item for item in x if item.isalpha()
                                                                    and item not in stop_words]))
    # Make bigram list of words
    dataframe['bigrams'] = dataframe['allwordsorig'].map(bigrams)
    # Make trigram list of words
    dataframe['trigrams'] = dataframe['allwordsorig'].map(trigrams)
    # Get all the pos tagged words in a single list for each document
    dataframe['poslist'] = dataframe['pos'].apply(lambda x: [item for sublist in x for item in sublist])
    # Calculate the frequency of each pos tagged word
    dataframe['mfreqpos'] = dataframe['poslist'].apply(nltk.FreqDist)
    # Get the stems of all the words
    dataframe['stemwordstuple'] = dataframe['allwords'].apply(lambda x: [(pstemmer.stem(item), item) for item in x])
    # Calculate frequency of stemmed words
    dataframe['mfreqstem'] = dataframe['stemwordstuple'].apply(nltk.FreqDist)
    # Calculate frequency of bigrams
    dataframe['mfreqbigrams'] = dataframe['bigrams'].apply(nltk.FreqDist)
    # Calculate frequency of trigrams
    dataframe['mfreqtrigrams'] = dataframe['trigrams'].apply(nltk.FreqDist)
    return dataframe


# Score documents based on pos - should be most exact match
def scoringpos(dataframe, list, poslist):
    for (w1, t1) in poskeywords:
        for idx, row in dataframe.iterrows():
            if (w1, t1) in row['poslist']:
                if not row['document'] in list[w1.lower()]:
                    list[w1.lower()].append(row['document'])
                    poslist[(w1.lower(), t1)].append(row['document'])
                    dataframe.loc[idx, 'score'] += row['mfreqpos'][(w1, t1)]
                    print('scored ' + str(row['mfreqpos'][(w1, t1)]) + ' for ' + str((w1, t1)) + ' in ' + row['document'])
            if w1.lower() in [x[0].lower() for x in row['poslist']]:
                for word, tag in row['poslist']:
                    if word.lower() == w1.lower() and (word.lower(), tag) not in poslist[word.lower(), tag]:
                        if not row['document'] in poslist[(word.lower(), tag)]:
                            poslist[(word.lower(), tag)].append(row['document'])
                            list[word.lower()].append(row['document'])
                            dataframe.loc[idx, 'score'] += (row['mfreqpos'][word.lower()] * 0.75)
                            print('scored ' + str(row['mfreqpos'][(word, tag)] * 0.75) + ' for ' + word + ' with ' + tag + ' in ' + row['document'])
    return dataframe


# Score documents based on stemmed words in cleansed dataset - so should discount stopwords and be sensible
def scoringstem(dataframe, list):
    for stem, word in stemkeywords:
        for idx, row in dataframe.iterrows():
            for s1, w1 in row['stemwordstuple']:
                if stem == s1 and w1 not in list[w1]:
                    if not row['document'] in list[w1]:
                        list[w1].append(row['document'])
                        dataframe.loc[idx, 'score'] += (row['mfreqstem'][(stem, w1)] * 0.5)
                        print('scored ' + str(row['mfreqstem'][(stem, w1)] * 0.5) + ' for ' + stem + ' - ' + word + ' and match was ' + s1 + ' - ' + w1 + ' in ' + row['document'])
    return dataframe


# Scoring for bigrams - exact match only
def scoringBigrams(dataframe, list):
    for word in keywords_bigrams:
        for idx, row in dataframe.iterrows():
            match = tuple(word.split())
            if match in row['bigrams']:
                if not row['document'] in list[match]:
                    list[match].append(row['document'])
                    dataframe.loc[idx, 'score'] += row['mfreqbigrams'][match]
    return dataframe

# Scoring for trigrams - exact match only
def scoringTrigrams(dataframe, list):
    for word in keywords_trigrams:
        for idx, row in dataframe.iterrows():
            match = tuple(word.split())
            if match in row['trigrams']:
                if not row['document'] in list[match]:
                    list[match].append(row['document'])
                    dataframe.loc[idx, 'score'] += row['mfreqtrigrams'][match]
    return dataframe


# Find keywords using POS, show the sentence the word was found in
def contextkeywords(dataframe):
    pdf.set_font('DejaVuSans-Bold', '', 12)
    pdf.cell(w=0,txt="Here are the exact keyword matches in context: ", ln=1, align="L")
    pdf.ln(10)
    for (w1, t1) in poskeywords:
        for idx, row in dataframe.iterrows():
            for index, r in enumerate(row['pos']):
                if (w1, t1) in r:
                    pdf.set_font('DejaVuSans-Bold', '', 10)
                    pdf.multi_cell(w=0, h=10, txt=row['document'] + ' - ', align="L")
                    pdf.set_font('DejaVu', '', 10)
                    pdf.multi_cell(w=0, h=10, txt=' '.join(row['words'][index]),  align="L")
                    pdf.ln(5)
    return dataframe


# Show all the documents that had keyword matches, for each keyword
def printkeywordmatches(list):
    pdf.set_font('DejaVuSans-Bold', '', 12)
    pdf.cell(w=0, txt="Keyword match results: ", ln=1, align="L")
    pdf.ln(10)
    for key, val in list.items():
        pdf.set_font('DejaVuSans-Bold', '', 10)
        pdf.multi_cell(w=0, h=10, txt="Documents containing keyword: " + key, align="L")
        pdf.ln(5)
        pdf.set_font('DejaVu', '', 10)
        pdf.multi_cell(w=0, h=10, txt=', '.join(val), align="L")
        pdf.ln(10)


# Show all the documents that had bigram keyword matches, for each keyword
def printkeywordmatchesBigrams(list):
    pdf.set_font('DejaVuSans-Bold', '', 12)
    pdf.cell(w=0, txt="Two-worded keyword match results: ", ln=1, align="L")
    pdf.ln(10)
    for key, val in list.items():
        pdf.set_font('DejaVuSans-Bold', '', 10)
        pdf.multi_cell(w=0, h=10, txt="Documents containing keywords: " + ' '.join(key), align="L")
        pdf.ln(5)
        pdf.set_font('DejaVu', '', 10)
        pdf.multi_cell(w=0, h=10, txt=', '.join(val), align="L")
        pdf.ln(10)


# Show all the documents that had trigram keyword matches, for each keyword
def printkeywordmatchesTrigrams(list):
    pdf.set_font('DejaVuSans-Bold', '', 12)
    pdf.cell(w=0, txt="Three-worded keyword match results: ", ln=1, align="L")
    pdf.ln(10)
    for key, val in list.items():
        pdf.set_font('DejaVuSans-Bold', '', 10)
        pdf.multi_cell(w=0, h=10, txt="Documents containing keywords: " + ' '.join(key), align="L")
        pdf.ln(5)
        pdf.set_font('DejaVu', '', 10)
        pdf.multi_cell(w=0, h=10, txt=', '.join(val), align="L")
        pdf.ln(10)


# tokenize each word in the text and then filter out non alphabet words, then get all the stems
def tokenize_and_stem(text):
    tokens = [word for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
    filtered_tokens = []
    for token in tokens:
        if re.search('[a-zA-Z]', token):
            filtered_tokens.append(token)
    stems = [pstemmer.stem(t) for t in filtered_tokens]
    return stems


# Main loop function
# Iterate over all files in the folder and process each one in turn

print('Starting processing - the following files have been processed:')
for input_file in glob.glob(os.path.join(input_path, '*.*')):
    # Grab the file name
    filename = os.path.basename(input_file)
 #   fname = os.path.splitext(filename)[0]
    print(filename)

    # Parse the file to get to the text
    parsed = parsewithtika(input_file)

    # Language detection algorithm is non - deterministic, which means that if you try to run it on a text which is
    # either too short or too ambiguous, you might get different results every time you run it
    if filterlanguage(parsed):
        continue

    # Ignore any documents with <100 words
    if len(parsed) < 100:
        continue

    # Sentence fragments
    sentences = sent_tokenize(parsed)

    # Build up dataframe
    temp = pd.Series([filename, sentences])
    d = d.append(temp, ignore_index=True)




print('\n')
d.reset_index(drop=True, inplace=True)
d.columns = ['document', 'sentences']


# Word tokenize the sentences, cleanup, parts of speech tagging
wordtokens(d)
d['score'] = 0

# Now we score in a calculated manner:
# Score 1 for matching word (case sensitive and POS), Score 0.75 for matching word (case insensitive,  stop words removed)
scoringpos(d, word_matches, pos_matches)
# Score 0.5 for matching stem of word (case insensitive, stop words removed)
scoringstem(d, word_matches)
# Score 1 for matching a bigram
scoringBigrams(d, bigram_matches)
# Score 1 for matching a trigram
scoringTrigrams(d, trigram_matches)


# Sort by scoring
d = d.sort_values('score', ascending=False)

# TODO: make this better - separate documents?

pdf = buildPDF()

# Print out the results of exact keyword matching
printkeywordmatches(word_matches)
printkeywordmatchesBigrams(bigram_matches)
printkeywordmatchesTrigrams(trigram_matches)


# Find words in context with POS
# TODO: Make this better
# TODO: show stem matches
# contextkeywords(d)

# Print sorted documents
print('\n')
pdf.ln(10)
pdf.set_font('DejaVuSans-Bold', '', 12)
pdf.cell(w=0,txt="Here are the scores based on cleansed data: ", ln=1, align="L")
pdf.ln(5)
pdf.set_font('DejaVu', '', 10)
print(d[['document', 'score']])

# Effective page width, or just epw
epw = pdf.w - 2 * pdf.l_margin

# Set column width to 1/4 of effective page width to distribute content
# evenly across table and page
col_width = epw / 2

# Text height is the same as current font size
th = pdf.font_size
data = d[['document', 'score']].values

for row in data:
    for datum in row:
        # Enter data in colums
        # Notice the use of the function str to coerce any input to the
        # string type. This is needed
        # since pyFPDF expects a string, not a number.
        pdf.cell(col_width, th, str(datum), border=1, align='C')
    pdf.ln(th)


# pdf.multi_cell(w=0, h=10, txt=d[['document', 'score']].to_string(index=False), align="L")
pdf.ln(10)

# cater for small no of docs
# cater for 0 scores
print(len(d))
if len(d) < 5:
    topdocs = d
else:
    topdocs = d.head(int(len(d)*0.1))

print(topdocs)

# Print results of NER for people
pdf.set_font('DejaVuSans-Bold', '', 12)
pdf.multi_cell(w=0, h=10, txt='People discovered:', align="L")
pdf.set_font('DejaVu', '', 10)
pdf.ln(5)
for doc in topdocs['ner']:
    for (a,b) in doc:
        if b == 'PERSON':
            pdf.multi_cell(w=0, h=10, txt=a, align="L")
pdf.ln(10)

# Print results of NER for organisations
pdf.set_font('DejaVuSans-Bold', '', 12)
pdf.multi_cell(w=0, h=0, txt='Orgs discovered:', align="L", border=1)
pdf.set_font('DejaVu', '', 10)
pdf.ln(5)
for doc in topdocs['ner']:
    for (a,b) in doc:
        if b == 'ORG':
            pdf.multi_cell(w=0, h=10, txt=a, align="L")

# Output the case document with all the printed results to PDF
pdf.output(output_path)