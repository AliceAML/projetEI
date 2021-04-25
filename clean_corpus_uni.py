"""
Clean corpus "Universités" using Beautiful soup
"""

from bs4 import BeautifulSoup
import os

SOURCE = "corpus/universites/"
RESULT = "corpus/uni_clean/"

num_files = len(os.listdir(SOURCE))

print(f"{num_files} fichiers à nettoyer.")

cleaned = 0

for i, file in enumerate(os.listdir(SOURCE)):
    try:
        with open(SOURCE + file) as html:
            soup = BeautifulSoup(html, "lxml")

        text = soup.get_text(strip=False, separator="\n")
        with open(RESULT + file, "w") as f:
            f.write(text)
        print(f"{i}/{num_files} nettoyé")
        cleaned += 1
    except Exception as e:
        print(f"{i}/{num_files} {e} - {file}")

print(f"{cleaned} fichiers sur {num_files} nettoyés.")