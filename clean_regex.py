import regex as re
name = "corpus_maison/Causette-84-bd-Ecriture inclusive.txt"
def load_file(filename):
    data = open(filename, 'r', encoding="utf-8")
    text = data.read()
    clean = text.replace("\t"," ").replace("\n", " ")
    output = clean.lower()

    data.close()

    return output


text = load_file(name)
sentences = re.split(r"(?<=\.|\?|\!)\s", text)

print("text 100",text[0:100])
# à propos de la regex : \b pour identifier les fins de mots dans une string (utilisable sur une pharse, il faut qu'il y ai
# un séparateur. si on passe sur des items : pas de sep)
print("sentences 10", sentences[:10])
def labelling(sentences) :
    out = []
    for one_sentence in sentences : 
        tempwords = one_sentence.split(' ')
        tempgold = []
        for index, word in enumerate(tempwords) :
            if re.search('·', word):
                tempgold.append(index)
        tempdict = dict()
        tempdict['words'] = tempwords
        tempdict['gold'] = tempgold
        out.append(tempdict)
    return out

labels = labelling(sentences)
for i in labels :
    if i['gold'] :
        for j in i['gold'] :
            print(i['words'][j])
          



# output = list(map(lambda x: x.replace('\t','').replace('\n',''),text))

# 1  nettoyer le texte (tabulation? passages de lignes?)
#       phrases coupees par des sauts de lignes = ok
#       phrases de titres : elles n'ont pas de points, ça les fusionne
# 2  mettre en forme de corpus (par textes entiers?)
# 3  estimer les meilleures regex

# \n([a-zA-Z0-9]+|[A-Z]\+[a-zA-Z0-9]+)\n
