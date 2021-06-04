import conllu

test = """# sent_id = 248
# text_no_ei =  Combien de fois montre-t-on l ’ ertzaina qui charge contre des manifestants ?
id	form	lemma	upostag	xpostag	feats	head	deprel	deps	misc
0	Combien	combien	ADV	ADV__PronType=Int	_	4	advmod	_	_
1	de	de	ADP	ADP	_	4	case	_	_
2	fois	fois	NOUN	NOUN__Gender=Fem|Number=Sing	_	4	nummod	_	_
3	montre-t-on	montre-t-on	ADV	ADV	_	6	amod	_	_
4	l	l	NOUN	NOUN	_	6	amod	_	_
5	’	’	PROPN	PROPN	_	0	ROOT	_	_
6	ertzaina	ertzaina	PROPN	PROPN	_	6	flat:name	_	_
7	qui	qui	PRON	PRON__PronType=Rel	_	9	nsubj	_	_
8	charge	charger	VERB	VERB__Mood=Ind|Number=Sing|Person=3|Tense=Pres|VerbForm=Fin	_	6	acl:relcl	_	_
9	contre	contre	ADP	ADP	_	12	case	_	_
10	des	un	DET	DET__Definite=Ind|Number=Plur|PronType=Art	_	12	det	_	_
11	manifestants	manifestant	NOUN	NOUN__Gender=Masc|Number=Plur	_	9	obl:arg	_	ei=manifestant-e-s
12	?	?	PUNCT	PUNCT	_	6	punct	_	_"""

data = conllu.parse(test)
print(data)