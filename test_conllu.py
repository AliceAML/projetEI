import conllu

test = """# sent_id = 12
# text_no_ei =  J ’ étais vraiment seul .
0	J	j	NOUN	NOUN	_	2	nummod	_	_
1	’	’	PUNCT	PUNCT	_	3	nsubj	_	_
2	étais	étai	ADJ	ADJ__Gender=Masc	_	5	cop	_	_
3	vraiment	vraiment	ADV	ADV	_	5	advmod	_	_
4	seul	seul	ADJ	ADJ__Gender=Masc|Number=Sing	_	0	ROOT	_	ei=seul·e
5	.	.	PUNCT	PUNCT	_	5	punct	_	_"""

data = conllu.parse(test)
print(data)