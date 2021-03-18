import regex as re
# data =  open("corpus_maison/Causette-84-bd-Ecriture inclusive.txt", "r", encoding="utf-8")
name = "corpus_maison/Causette-84-bd-Ecriture inclusive.txt"
def load_file(filename):
    data = open(filename, 'r', encoding="utf-8")
    text = data.read()
    clean = text.replace("\t"," ").replace("\n", " ")
    output = clean.lower()

    data.close()

    return output


text = load_file(name)
# sentences = re.split(r"[\s]+[\.?!]", text)
# sentences = re.split(r"\s", text)
# sentences = re.split(r"\s", text)
# sentences = re.split(r"\s", text)
# sentences = re.split(r"(\.|(\s+(\?|!))\s)", text)
# sentences = re.findall(r'(?:\d[,.]|[^,.])*(?:[,.]|$)', text)
sentences = re.split(r"(?<=\.|\?|\!)\s", text)

print("text 100",text[0:100])
# à propos de la regex : \b pour check les fins de mots dans une string (il faut qu'il y ai
# un séparateur. si on passe sur des items : pas de sep)
print("sentences 10", sentences[:10])
out = []
for one_sentence in sentences : 
    tempwords = one_sentence.split(' ')
    tempgold = []
    for word in tempwords :
        if re.search('·', word):
            tempgold.append(True)
        else :
            tempgold.append(False)
    tempdict = dict()
    tempdict['words'] = tempwords
    tempdict['gold'] = tempgold
    out.append(tempdict)

print(out)
          



# output = list(map(lambda x: x.replace('\t','').replace('\n',''),text))

# 1  nettoyer le texte (tabulation? passages de lignes?)
#     phrases coupees par des sauts de lignes
# 2  mettre en forme de corpus (par textes entiers?)
# 3  estimer les meilleures regex

# \n([a-zA-Z0-9]+|[A-Z]\+[a-zA-Z0-9]+)\n