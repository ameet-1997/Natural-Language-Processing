topic_mapping = {
    'alt.atheism': 0,
    'comp.graphics': 1,
    'comp.os.ms-windows.misc': 2,
    'comp.sys.ibm.pc.hardware': 3,
    'comp.sys.mac.hardware': 4,
    'misc.forsale': 5,
    'rec.autos': 6,
    'rec.motorcycles': 7,
    'rec.sport.baseball': 8,
    'rec.sport.hockey': 9,
    'sci.crypt': 10,
    'sci.electronics': 11,
    'sci.med': 12,
    'sci.space': 13,
    'soc.religion.christian': 14,
    'talk.politics.guns': 15,
    'talk.politics.mideast': 16,
    'talk.politics.misc': 17,
    'talk.religion.misc': 18
    }


# Mapping of indices to topic name
inverse_mapping = {
    0 : 'alt.atheism',
    1 : 'comp.graphics',
    2 : 'comp.os.ms-windows.misc',
    3 : 'comp.sys.ibm.pc.hardware',
    4 : 'comp.sys.mac.hardware',
    5 : 'misc.forsale',
    6 : 'rec.autos',
    7 : 'rec.motorcycles',
    8 : 'rec.sport.baseball',
    9 : 'rec.sport.hockey',
    10 : 'sci.crypt',
    11 : 'sci.electronics',
    12 : 'sci.med',
    13 : 'sci.space',
    14 : 'soc.religion.christian',
    15 : 'talk.politics.guns',
    16 : 'talk.politics.mideast',
    17 : 'talk.politics.misc',
    18 : 'talk.religion.misc'
    }

# Stopwords
# Includes words which are not present in nltk.corpus.stopwords('english')
stop = ['a', 'about', 'above', 'after', 'again', 'against', 'all', 'am', 'an', 'and', 'any', 'are', "aren't", 'as', 'at', 'be', 'because', 'been', 'before', 'being', 'below', 'between', 'both', 'but', 'by', "can't", 'cannot', 'could', "couldn't", 'did', "didn't", 'do', 'does', "doesn't", 'doing', "don't", 'down', 'during', 'each', 'few', 'for', 'from', 'further', 'had', "hadn't", 'has', "hasn't", 'have', "haven't", 'having', 'he', "he'd", "he'll", "he's", 'her', 'here', "here's", 'hers', 'herself', 'him', 'himself', 'his', 'how', "how's", 'i', "i'd", "i'll", "i'm", "i've", 'if', 'in', 'into', 'is', "isn't", 'it', "it's", 'its', 'itself', "let's", 'me', 'more', 'most', "mustn't", 'my', 'myself', 'no', 'nor', 'not', 'of', 'off', 'on', 'once', 'only', 'or', 'other', 'ought', 'our', 'ours', 'ourselves', 'out', 'over', 'own', 'same', "shan't", 'she', "she'd", "she'll", "she's", 'should', "shouldn't", 'so', 'some', 'such', 'than', 'that', "that's", 'the', 'their', 'theirs', 'them', 'themselves', 'then', 'there', "there's", 'these', 'they', "they'd", "they'll", "they're", "they've", 'this', 'those', 'through', 'to', 'too', 'under', 'until', 'up', 'very', 'was', "wasn't", 'we', "we'd", "we'll", "we're", "we've", 'were', "weren't", 'what', "what's", 'when', "when's", 'where', "where's", 'which', 'while', 'who', "who's", 'whom', 'why', "why's", 'with', "won't", 'would', "wouldn't", 'you', "you'd", "you'll", "you're", "you've", 'your', 'yours', 'yourself', 'yourselves']
stop.extend((str('subject'), str('from'), str('organization'
            ), str('organisation')))

cats = ['alt.atheism',
     'comp.graphics',
     'comp.os.ms-windows.misc',
     'comp.sys.ibm.pc.hardware',
     'comp.sys.mac.hardware',
     'misc.forsale',
     'rec.autos',
     'rec.motorcycles',
     'rec.sport.baseball',
     'rec.sport.hockey',
     'sci.crypt',
     'sci.electronics',
     'sci.med',
     'sci.space',
     'soc.religion.christian',
     'talk.politics.guns',
     'talk.politics.mideast',
     'talk.politics.misc',
     'talk.religion.misc']


# Leaf to topic mapping
leaf_to_topic = {
    7 : 0,
    8 : 14,
    9 : 18,
    10 : 1,
    11 : 2,
    12 : 3,
    13 : 4,
    14 : 8,
    15 : 9,
    16 : 15,
    17 : 16,
    18 : 17,
    19 : 6,
    20 : 7,
    21 : 10,
    22 : 12,
    23 : 11,
    24 : 13,
    0 : 5       # Misc For Sale
}

inverse_leaf_to_topic = {
    0 : 7,
    14 : 8,
    18 : 9,
    1 : 10,
    2 : 11,
    3 : 12,
    4 : 13,
    8 : 14,
    9 : 15,
    15 : 16,
    16 : 17,
    17 : 18,
    6 : 19,
    7 : 20,
    10 : 21,
    12 : 22,
    11 : 23,
    13 : 24,
    5 : 0
}

# inverse_leaf_to_topic = {
#     0 : 7,
#     15 : 8,
#     19 : 9,
#     1 : 10,
#     2 : 11,
#     3 : 12,
#     4 : 13,
#     9 : 14,
#     10 : 15,
#     17 :16,
#     18 : 17,
#     16 : 18,
#     7 : 19,
#     8 : 20,
#     11 : 21,
#     13 : 22,
#     12 : 23,
#     14 : 24,
#     6 : 0
# } 

# Mapping of indices in dataset to topics
# topic_mapping = {
#     'alt.atheism': 0,
#     'comp.graphics': 1,
#     'comp.os.ms-windows.misc': 2,
#     'comp.sys.ibm.pc.hardware': 3,
#     'comp.sys.mac.hardware': 4,
#     'comp.windows.x': 5,
#     'misc.forsale': 6,
#     'rec.autos': 7,
#     'rec.motorcycles': 8,
#     'rec.sport.baseball': 9,
#     'rec.sport.hockey': 10,
#     'sci.crypt': 11,
#     'sci.electronics': 12,
#     'sci.med': 13,
#     'sci.space': 14,
#     'soc.religion.christian': 15,
#     'talk.politics.misc': 16,
#     'talk.politics.guns': 17,
#     'talk.politics.mideast': 18,
#     'talk.religion.misc': 19
#     }

# # Mapping of indices to topic name
# inverse_mapping = {
#     0 : 'alt.atheism',
#     1 : 'comp.graphics',
#     2 : 'comp.os.ms-windows.misc',
#     3 : 'comp.sys.ibm.pc.hardware',
#     4 : 'comp.sys.mac.hardware',
#     5 : 'comp.windows.x',
#     6 : 'misc.forsale',
#     7 : 'rec.autos',
#     8 : 'rec.motorcycles',
#     9 : 'rec.sport.baseball',
#     10 : 'rec.sport.hockey',
#     11 : 'sci.crypt',
#     12 : 'sci.electronics',
#     13 : 'sci.med',
#     14 : 'sci.space',
#     15 : 'soc.religion.christian',
#     16 : 'talk.politics.misc',
#     17 : 'talk.politics.guns',
#     18 : 'talk.politics.mideast',
#     19 : 'talk.religion.misc'
#     }