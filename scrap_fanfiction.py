import urllib.request, urllib.error, urllib.parse
from lxml.html import fromstring

"""Script modifié à à partir d'un script de Guillaume Wisniewski"""

urlprefix = "https://www.fanfiction.net/s/{}/{}"

ids = [, ]

for id_article in range(212, 1796):
    out = "-> ID_ARTICLE {} / 1795"
    print(out.format(id_article))

    try:
        response = urllib.request.urlopen(urlprefix.format(id_article))
        html = response.read()
        dom = fromstring(html.decode("utf-8"))
        sels = dom.xpath('//div[(@class="texte-brochure")]//*')
    except Exception as e:
        print(e)
        continue

    with open(f"corpus_maison/corpus_infokiosque/infokiosque_{id_article}", "wt", encoding="utf-8") as f:
        for paragraph in sels:
            if paragraph.text:
                f.write(paragraph.text +" ")