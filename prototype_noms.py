# mini prototype qui prend des noms masculins sg et renvoie une version "inclusive"
# par ex makeEI("chanteur") renvoie "chanteur·euse"
# développé avec le livre "Manuel de grammaire non sexiste et inclusive", pp.41-

# amélioration à faire : 
#   accepter mots au pluriel et au féminin
#   ne pas modifier les mots épicènes (ex: violoniste)
#   gérer les mots comme "chanteur" qui finissent par "teur" mais n'ont pas de féminin en "trice"


esse = {"abbé" : "abbesse", #p.43-44 du livre
    "âne" : "ânesse",
    "borgne" : "borgnesse",
    "bougre" : "bougresse",
    "buffle" : "bufflesse",
    "centaure" : "centauresse",
    "chanoine" : "chanoinesse",
    "chef" : "cheffesse",
    "clown" : "clownesse",
    "doge" : "dogaresse",
    "drôle" : "drôlesse",
    "druide" : "druidesse",
    "duc" : "duchesse",
    "faune" : "faunesse",
    "gonze" : "gonzesse",
    "hôte" : "hôtesse",
    "ivrogne" : "ivrognesse",
    "maire" : "mairesse",
    "pair" : "pairesse",
    "pape" : "papesse",
    "pauvre" : "pauvresse",
    "peintre" : "peintresse",
    "poète" : "poétesse",
    "prêtre" : "prêtresse",
    "prince" : "princesse",
    "prophète" : "prothétesse",
    "comte" : "comtesse",
    "maître" : "maîtresse",
    "sauvage" : "sauvagesse",
    "consule" : "consulesse",
    "moine" : "moinesse",
    "suisse" : "suissesse",
    "contremaître" : "contremaîtresse",
    "mulâtre" : "mulâtresse",
    "tigre" : "tigresse",
    "nègre" : "négresse",
    "traître" : "traîtresse",
    "devin" : "devineresse",
    "notaire" : "notairesse",
    "type" : "typesse",
    "diable" : "diablesse",
    "ogre" : "ogresse",
    "vicomte" : "vicomtesse"
    }

exceptions = { # p.44/45 du livre
    "canard" : "cane",
    "canut" : "canuse",
    "chef" : "cheftaine",
    "chevreuil" : "chevrette",
    "compagnon" : "compagne",
    "daim" : "daine",
    "diacre" : "diaconesse",
    "dieu" : "déesse",
    "dindon" : "dinde",
    "empereur" : "impératrice",
    "favori" : "favorite",
    "fils" : "fille",
    "héros" : "héroïne",
    "lévrier" : "levrette",
    "loup" : "louve",
    "loup-cervier" : "loup-cerve",
    "merle" : "merlette",
    "mulet" : "mule",
    "neveu" : "nièce",
    "péquenot" : "péquenaude",
    "perroquet" : "perruche",
    "pierrot" : "pierrette",
    "rousseau" : "rousse",
    "roi" : "reine",
    "salaud" : "salope",
    "serviteur" : "servante",
    "sphinx" : "sphinge",
    "sylphe" : "sylphide",
    "tsar" : "tsarine",
    "amant" : "maîtresse" ,
    "bélier" : "brebis",
    "bouc" : "chèvre" ,
    "cerf" : "biche",
    "chien de chasse" : "lice",
    "coq" : "poule",
    "confrère" : "consœur",
    "étalon" : "jument" ,
    "frère" : "sœur",
    "frérot" : "sœurette",
    "garçonnet" : "fillette",
    "garçon" : "fille",
    "gars" : "fille",
    "gendre" : "bru",
    "hébreu" : "juive",
    "homme" : "femme",
    "jars" : "oie",
    "lièvre" : "hase",
    "lord" : "lady",
    "mâle" : "femelle",
    "mari" : "femme" ,
    "matou" : "chatte" ,
    "monsieur" : "madame",
    "oncle" : "tante",
    "papa" : "maman",
    "parrain" : "marraine",
    "père" : "mère",
    "sanglier" : "laie",
    "seigneur" : "dame",
    "singe" : "guenon",
    "taureau" : "vache",
    "valet de chambre" : "femme de chambre",
    "verrat" : "truie" 
}

def makeEI(nom, point="·"):
    radical = nom[0] + nom[1:].lower() #je met tout sauf la première lettre en minuscule
    fem_exposant = "e" # possibilité de le faire à la toute fin ?
    
    if radical.endswith("e"): # tentative très brutale de traiter les mots épicènes (ex: violoniste)
        return radical

    elif radical in exceptions:
        fem_exposant = exceptions[radical]
        point = "/" # si deux formes totalement différentes, on peut utiliser un slash ?

    elif radical in esse:
        fem_exposant = "sse"

    elif radical.endswith(("el", "en", "on", "et")): # si on a une de ces terminaisons
        fem_exposant = radical[-1] + "e" # on ajoute la dernière lettre à l'exposant féminin + le e (on pourrait juste ajouter le e tout à la fin aussi)
    elif radical.endswith("er"):
        fem_exposant = "ère"
    elif radical.endswith("x"):
        fem_exposant = "se"
    elif radical.endswith("eau"):
        fem_exposant = "elle"
        point = "/" # optionnel, à discuter... mais c'est intéressant de voir qu'on peut choisir de modifier le séparateur
    elif radical.endswith("f"):
        fem_exposant = "ve"
    elif radical.endswith("c"):
        fem_exposant = "que"
    elif radical.endswith("teur"): # on voit sur ces deux règles (eur/teur) que l'ordre est important
        fem_exposant = "trice"
    elif radical.endswith("eur"):
        fem_exposant = "euse"
    return radical + point + fem_exposant # règle la plus générale, p.41

def printEI(nom):
    print(nom, "->", makeEI(nom))



# TESTs pris dans le livre
    # # test règle de base p.41
    # print("***tests p.41***")
    # printEI("ami")
    # printEI("ours")
    # printEI("intellectuel")
    # printEI("chameau")
    # printEI("Italien")
    # printEI("vigneron")
    # printEI("cadet")
    # printEI("conseiller")
    # printEI("amoureux")

    # print("\n***tests p.42")
    # printEI("veuf")
    # printEI("turc")
    # printEI("voleur")
    # printEI("facteur")
    # printEI("assassin")
    # printEI("sultan")
    # printEI("châtelain")

    # print("\n***tests p.43")
    # printEI("avocat")
    # printEI("idiot")
    # printEI("bourgeois")

    # for exc in esse.keys():
    #     printEI(exc)

    # print("\n*** tests exceptions")
    # for exc in exceptions.keys():
    #     printEI(exc)

# petite interface pour que l'utilisateur·ice puisse rentrer ses propres mots

print("""Bienvenue dans ce petit prototype d'assistant d'EI très très basique.\n
Attention, ce programme fait encore beaucoup d'erreurs !\n""")

user_input = input("Rentre un nom commun au masculin singulier afin de voir une version inclusive : \n\t")
printEI(user_input)

while True:
    user_input = input("Rentre un autre mot : \n\t")
    printEI(user_input)