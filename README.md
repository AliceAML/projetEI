# Projet de M1 d'Alice HAMMEL et Marjolaine RAY : automatiser l'écriture inclusive
Juin 2021

---------------------------------------------------------------
Projet de détection automatique de formes à écrire en écriture inclusive dans un texte et de transformation de ces formes.
---------------------------------------------------------------

1) Dézipper le projet
2) Ouvrir le terminal et se placer dans le répertoire du projet

3) Nous recommandons d'utiliser un environnement virtuel et d'y installer les dépendances de ce projet :
    ```
    pip install virtualenv
    python3 -m venv env
    source env/bin/activate
    pip install -r requirements.txt
    ```

Pour sortir de l'environnement virtuel après utilisation du programme :

    deactivate

Il est particulièrement important de ne pas utiliser spaCy 3, car le module spacy-conll n'est que compatible avec spaCy 2.

4) Le modèle `fr-core-news-sm` de spaCy doit être téléchargé :
    
    `python3 -m spacy download fr_core_news_sm`

5) Le programme principal model_ei.py permet la conversion de phrases en écriture inclusive à partir de modèles, ainsi que l'entraînement et l'évaluation de ces modèles.

Il doit être exécuté depuis le répertoire parent :

    python3 codeEI/model_ei.py

il dispose d'une aide en ligne qui détaille son usage :

    python3 codeEI/model_ei.py --help

Deux modèles déjà entraînés peuvent être chargés :
- Pipeline_V2 (modèle SVM par défaut)
- RandomForestClassifier_V4
