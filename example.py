from Db import Db
import numpy as np
import sklearn.cluster
import distance
import re

def getWordList():
    db = Db('loe3','root','youwonder.')
    wordlist = db.database('MetalArchives').table('Artist').select('distinct lyricalThemes').get()
    uniqueWords = []
    for row in wordlist:
        words = row[0].split(',')
        for word in words:
            testWord = re.sub('\s\(.*?\)','',word).strip().lower()
            if testWord not in uniqueWords:
                uniqueWords.append(testWord)
    return uniqueWords

def doClustering(wordList):
    words = np.asarray(wordList) #So that indexing with a list will work
    lev_similarity = -1*np.array([[distance.levenshtein(w1,w2) for w1 in wordList] for w2 in wordList])
    affprop = sklearn.cluster.AffinityPropagation(affinity="precomputed", damping=.5,random_state=0)
    affprop.fit(lev_similarity)
    for cluster_id in np.unique(affprop.labels_):
        exemplar = words[affprop.cluster_centers_indices_[cluster_id]]
        cluster = np.unique(words[np.nonzero(affprop.labels_==cluster_id)])
        cluster_str = ", ".join(cluster)
        print(" - *%s:* %s" % (exemplar, cluster_str))

def getClusterCenters(wordList):
    cluster_centers = []
    words = np.asarray(wordList) #So that indexing with a list will work
    lev_similarity = -1*np.array([[distance.levenshtein(w1,w2) for w1 in wordList] for w2 in wordList])
    print(lev_similarity)
    affprop = sklearn.cluster.AffinityPropagation(affinity="precomputed", damping=.5,random_state=0)
    affprop.fit(lev_similarity)
    for cluster_id in np.unique(affprop.labels_):
        exemplar = words[affprop.cluster_centers_indices_[cluster_id]]
        cluster_centers.append(exemplar)
    return cluster_centers

def getLyrics(songScriptUID):
    db = Db('loe3','root','youwonder.')
    result = db.database('LOE').table('SongScript').select('*').where("UID","=",songScriptUID).get()
    individualLyrics = re.sub('[\r\n]+',' ',result[0][2]).strip().split(' ')
    return individualLyrics


#print(getLyrics("91"))
#doClustering(getWordList())
#print(getClusterCenters(getWordList()))

lyricsToTest = getLyrics("91")
lyricalThemes = getClusterCenters(getWordList())
lev_similarity = -1*np.array([[distance.levenshtein(w1,w2) for w1 in lyricalThemes] for w2 in lyricsToTest])
print(lev_similarity)
