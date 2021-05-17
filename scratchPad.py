import numpy as np
import sklearn.cluster
import distance

words = "horror,blasphemy,gore,death,mutilation,insanity,murder,hate,anger,history,literature,war,mythology,society,religion,life,anti-fascism,satanism,paganism,mental disorders,philosophy,emotions,spirituality,inner struggles,anti-religion,pain,misery,politics,dragons,darkness,battles,evil,ancient ones,occultism,suffering,occult,corruption,satan,dark fantasy,rock/metal,sci-fi,fantasy,rituals,lovecraft,metal,perversion,paranoia,psychology,humor,anti-christianity,hatred,vikings,warfare,destruction,despair,romance,rebellion,human issues,violence,torture,cannibalism,zombies,apocalypse,misanthropy,loss,necrophilia,sex,drugs,nature,depression,winter,humanity,misogyny,rape,hell,tolkien,humour,social issues,environment,armageddon,fear,metaphysics,h.p. lovecraft,personal struggles,fishing,cosmos,esoteric,heritage,depravity,perversions,astral,sorrow,sadness,heavy metal,love,battle,tales,magic,chaos,thelema,revenge,n/a,heathenism,civilization,theology,human condition,madness,surrealism,doom,nihilism,mysticism,sickness,killing,isolation,devil worship,self-destruction,fiction,norse mythology,medieval,solitude,freedom,the occult,desecration,annihilation,disease,cosmology,serial killers,vengeance,instrumental,human nature,legends,corpses,macabre,suicide,social criticism,sick humour,space,existentialism,anxiety,dystopia,genocide,science fiction,aliens,personal experiences,thrash,physics,nuclear apocalypse,inner struggle,metaphysical,dreams,fetishism,nationalism,demons,social themes,existence,wars".split(",") #Replace this line
words = np.asarray(words) #So that indexing with a list will work
lev_similarity = -1*np.array([[distance.levenshtein(w1,w2) for w1 in words] for w2 in words])

clustering = sklearn.cluster.AffinityPropagation(affinity="precomputed", damping=.5,random_state=0).fit(lev_similarity)
for cluster_id in np.unique(clustering.labels_):
    exemplar = words[clustering.cluster_centers_indices_[cluster_id]]
    print(exemplar)
    #cluster = np.unique(words[np.nonzero(clustering.labels_==cluster_id)])
    #cluster_str = ", ".join(cluster)
    #print(" - *%s:* %s" % (exemplar, cluster_str))
