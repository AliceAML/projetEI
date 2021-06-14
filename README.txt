Projet de M1 d'Alice HAMMEL et Marjolaine RAY : automatiser l'écriture inclusive
Juin 2021

Les packages détaillés dans requirements.txt doivent être installés.

    pip install -r requirements.txt

Il est particulièrement important de ne pas utiliser spaCy 3, car le module spacy-conll n'est que compatible avec spaCy 2.

Le modèle fr-core-news-sm doit être téléchargé :
    python -m spacy download fr_core_news_sm

Le programme principal model_ei.py permet la conversion de phrases en écriture inclusive à partir de modèles, ainsi que l'entraînement et l'évaluation de ces modèles.

Il doit être exécuté depuis le répertoire parent :
    python3 codeEI/model_ei.py

il dispose d'une aide en ligne qui détaille son usage :
    python3 codeEI/model_ei.py --help