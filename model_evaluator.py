#IMPORT ALL LIBRARIES
import pandas as pd 
import numpy as np
import streamlit as st
from gsdmm import MovieGroupProcess
from sklearn.feature_extraction.text import CountVectorizer
from HanTa import HanoverTagger as ht
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords

#INTILIZE TAGGER
tagger = ht.HanoverTagger('morphmodel_ger.pgz')

#INITIALIZE GERMAN STOPWORDS
stop_words_de = list(set(stopwords.words('german')))
stop_words_de.extend(['Ja','Nein','KI','Künstlicher','künsticher','Künstliche','künstiche','Intelligenz','AI','ki','kI','Ai','aI','Artifizielle','artifizielle','intelligenz','darüber','Darüber','TheCityGame','verbessern','Jahr','App','Aktivität','Auswirkung','Idee','Dokument','Datum','einschließlich','Gruppen','Gruppe','profitieren','Produkte','Produkt','Projekt','geben','mindestens','daraus','open','mögen','Aktivitäten','Projekt','euro','tiber', 'direkt','altern','erheblich','wodurch','jedoch','schließlich','lokale','Informationen','Daten','Ziel','Bürgern','bereits','Unternehmen','verschieden','theCityGame','Auswirkungen','nachhaltig','Verbraucher','lokal','Plattform','neu','neue','Menschen','Mensch','Bürger','Zielgruppe','eben','soziale','sozial','verwenden','Personen','Person','gut','bieten','Benutzer','einfach','leicht','groß','Produkt ', 'Produkte ', 'Aktie ', 'Anteile ', 'Dokument ', 'Unterlagen ', 'myevent ', 'Ziel ', 'Ziele ', 'datum ', 'Datum ', 'schließlich ', 'Vorteil ', 'benötigen ', 'erfordert ', 'Sozial ', 'Täglich ', 'verwalten ', 'verwaltet ', 'echt ', 'existieren ', 'existiert ', 'klein ', 'datum ', 'Datum ', 'benötigen ', 'erfordert ', 'zusätzlich ', 'zusätzlich ', 'niedrig ', 'absichtlich ', 'Veranstaltung ', 'Veranstaltungen ', 'Einschlag ', 'Auswirkungen ', 'App ', 'Apps ', 'erstellen ', 'Verbraucher', 'Verbraucherin', 'Verbraucher ', 'schafft ', 'Aktivität ', 'Aktivitäten ', 'Region ', 'Regionen ', 'Aktion ', 'Aktionen ', 'sich beteiligen ', 'nimmt teil ', 'Bedienung ', 'Dienstleistungen ', 'Ziel ', 'Ziele ', 'Projekt ', 'Projekte ', 'Plattform ', 'Plattformen ', 'Geschichte ', 'Geschichten ', 'Leben ', 'lebte ', 'Kunde', 'Kundin', 'Kunden ', 'Bürger', 'Bürgerin', 'Bürger ', 'Nutzer', 'Nutzerin', 'Benutzer ', 'Gruppe ', 'Gruppen ', 'Menschen ', 'Person ', 'Personen ', 'brauchen ', 'Bedürfnisse ', 'wollen ', 'will ', 'könnten ', 'geben ', 'gibt ', 'Jahr ', 'Jahre ', 'Zeit ', 'mal ', 'Situation ', 'Situationen ', 'Gruppen ', 'öffnen ', 'Ideen ', 'Unterstützung ', 'Personen ', 'öffnet ', 'unterstützt ', 'gut ', 'führen ', 'führt ', 'Werkzeug ', 'Werkzeuge ', 'Kunde', 'Kundin', 'Verbraucher', 'Verbraucherin', 'Kunden ', 'Verbraucher ', 'Datum ', 'Bürger ', 'Geräte ', 'Individuell ', 'Einzelpersonen ', 'Prozess ', 'Spieler', 'Spielerin', 'Spieler', 'Spielerinnen', 'Nutzer', 'Nutzerin', 'Benutzer ', 'Aktion ', 'Aktionen ', 'Veränderung ', 'sich beteiligen ', 'Dienstleistungen ', 'Ziel ', 'Gruppe ', 'Gemeinschaft ', 'Bürger', 'Bürgerin', 'erstellen ', 'Anwendung ', 'Leben ', 'Person ', 'verbessern ', 'Öffentlichkeit ', 'Punkt ', 'Region ', 'Dokument ', 'Einschlag ', 'Jahr ', 'datum ', 'Bedienung ', 'lokal ', 'Gruppe ', 'Niveau ', 'aktivieren ', 'Nutzer', 'Nutzerin', 'zur Verfügung stellen ', 'Problem ', 'Plattform ', 'Taucto ', 'Menschen ', 'Projekt ', 'Unternehmen ', 'Firmen ', 'Entscheidung ', 'Entscheidungen ', 'Werkzeug ', 'Werkzeuge ', 'Niveau ', 'Ebenen ', 'Objekt ', 'Objekte ', 'Taucto ', 'Rohre ', 'Nutzer', 'Nutzerin', 'Benutzer ', 'erstellen ', 'schafft ', 'Ziel ', 'Ziele ', 'Projekt ', 'Projekte ', 'Gruppe ', 'Gruppen ', 'Bürger', 'Bürgerin', 'Bürger ', 'Gemeinschaft ', 'lokal ', 'Plattform ', 'Plattformen ', 'Personen ', 'Menschen ', 'Person ', 'zum ', 'kann ', 'könnte ', 'jedoch ', 'verwenden ', 'Verwendet ', 'machen ', 'macht ', 'gut ', 'Nutzer', 'Nutzerin', 'ebenfalls ', 'Direkte ', 'Idee ', 'von ', 'Gegenstand ', 'verwenden ', 'ein ', 'Über ', 'über ', 'nach ', 'nochmal ', 'gegen ', 'Auge ', 'alle ', 'bin ', 'ein ', 'und ', 'irgendein ', 'sind ', 'aren ', 'sind nicht ', 'wie ', 'beim ', 'Sein ', 'weil ', 'gewesen ', 'Vor ', 'Sein ', 'unten ', 'zwischen ', 'beide ', 'aber ', 'durch ', 'können ', 'konnte nicht ', 'konnte nicht ', 'd ', 'tat ', 'nicht ', 'nicht ', 'tun ', 'tut ', 'tut es nicht ', 'nicht ', 'tun ', 'Don ', 'nicht ', 'Nieder ', 'während ', 'jeder ', 'wenige ', 'zum ', 'von ', 'des Weiteren ', 'hätten ', 'hatte nicht ', 'hatte nicht ', 'hat ', 'Huh ', 'hat nicht ', 'haben ', 'Oase ', 'habe nicht ', 'haben ', 'mit ', 'ihm', 'ihr', 'Hier ', 'ihres ', 'Sie selber ', 'ihm ', 'selbst ', 'seine ', 'Wie ', 'ich ', 'wenn ', 'im ', 'in ', 'ist ', 'isn ', 'ist nicht ', 'es ', 'es ist ', 'es ist ', 'selbst ', 'gerade ', 'll ', 'm ', 'Pferd ', 'mich ', 'könnte nicht ', 'könnte nicht ', 'Mehr ', 'die meisten ', 'darf nicht ', 'darf nicht ', 'meine ', 'mich selber ', 'brauche nicht ', 'brauche nicht ', 'Nein ', 'Noch ', 'nicht ', 'jetzt ', 'Ö ', 'von ', 'aus ', 'auf ', 'Einmal ', 'nur ', 'oder ', 'andere ', 'unsere ', 'unsere ', 'uns selbst ', 'aus ', 'Über ', 'besitzen ', 'Hitze ', 's ', 'gleich ', 'Berg ', 'Shantou ', 'Schlange ', 'sie ist ', 'sollte ', 'sollte haben ', 'sollte nicht ', 'sollte nicht ', 'damit ', 'etwas ', 'eine solche ', 't ', 'als ', 'Das ', 'das wird ', 'das ', 'ihr ', 'ihre ', 'Sie ', 'sich ', 'dann ', 'Dort ', 'diese ', 'Sie ', 'diese ', 'jene ', 'durch ', 'zu ', 'auch ', 'unter ', 'bis um ', 'oben ', 'und ', 'sehr ', 'war ', 'warn ', 'war nicht ', 'wir ', 'wurden ', 'kein Zutritt ', 'waren nicht ', 'Was ', 'wann ', 'wo ', 'welche ', 'während ', 'Wer ', 'wem ', 'Warum ', 'werden ', 'mit ', 'gewonnen ', 'Gewohnheit ', 'würde nicht ', 'würde nicht ', 'und ', 'Haben ', 'du würdest ', 'du wirst ', 'du bist ', 'Sie ', 'Ihre ', 'deine ', 'du selber ', 'euch ', 'könnten ', 'er würde ', 'Hölle ', 'er ist ', 'hier ist ', 'wie ist ', 'Ich würde ', 'krank ', 'Ich bin ', 'Ich habe ', 'Lasst uns ', 'sollen ', 'Schuppen ', 'Schale ', 'das ist ', 'da ist ', 'Sie würden ', 'sie werden ', 'Sie sind ', 'Sie haben ', 'heiraten ', 'Gut ', 'wurden ', 'wir haben ', 'was ist ', 'wann ist ', 'wo ist ', 'wer ist ', 'warum ist ', 'würde ', 'imstande ', 'abst ', 'Übereinstimmung ', 'gemäß ', 'entsprechend ', 'über ', 'Handlung ', 'tatsächlich ', 'hinzugefügt ', 'adj ', 'betroffen ', 'beeinflussen ', 'betrifft ', 'danach ', 'Ah ', 'fast ', 'allein ', 'entlang ', 'bereits ', 'ebenfalls ', 'obwohl ', 'immer ', 'unter ', 'unter ', 'bekannt geben ', 'Ein weiterer ', 'irgendjemand ', 'jedenfalls ', 'nicht mehr ', 'jemand ', 'etwas ', 'wie auch immer ', 'Sowieso ', 'irgendwo ', 'offenbar ', 'etwa ', 'nicht ', 'entstehen ', 'um ', 'beiseite ', 'Fragen ', 'fragen ', 'auth ', 'verfügbar ', 'Weg ', 'schrecklich ', 'b ', 'zurück ', 'wurden ', 'werden ', 'wird ', 'Werden ', 'vorweg ', 'Start ', 'Anfang ', 'Anfänge ', 'beginnt ', 'hinter ', 'glauben ', 'neben ', 'Außerdem ', 'darüber hinaus ', 'Biol ', 'kurz ', 'kurz ', 'c ', 'Das ', 'kam ', 'kann nicht ', 'kippen ', 'Ursache ', 'Ursachen ', 'sicher ', 'bestimmt ', 'Was ', 'mit ', 'Kommen Sie ', 'kommt ', 'enthalten ', 'enthält ', 'enthält ', 'konnte nicht ', 'Datum ', 'anders ', 'getan ', 'nach unten ', 'fällig ', 'ist ', 'ed ', 'Quote ', 'bewirken ', 'z.B ', 'acht ', 'achtzig ', 'entweder ', 'sonst ', 'anderswo ', 'Ende ', 'Ende ', 'genug ', 'insbesondere ', 'und ', 'usw ', 'sogar ', 'je ', 'jeder ', 'jeder ', 'jeder ', 'alles ', 'überall ', 'Ex ', 'außer ', 'f ', 'weit ', 'ff ', 'fünfte ', 'zuerst ', 'fünf ', 'Fix ', 'gefolgt ', 'folgenden ', 'folgt ', 'ehemalige ', 'früher ', 'her ', 'gefunden ', 'vier ', 'Außerdem ', 'G ', 'gegeben ', 'bekommen ', 'bekommt ', 'bekommen ', 'geben ', 'gegeben ', 'gibt ', 'geben ', 'gehen ', 'geht ', 'Weg ', 'habe ', 'bekommen ', 'h ', 'das passiert ', 'kaum ', 'heiß ', 'daher ', 'Jenseits ', 'hiermit ', 'hierin ', 'hier ist ', 'hierauf ', 'er ist ', 'Hallo ', 'versteckt ', 'hierher ', 'Zuhause ', 'aber ', 'jedoch ', 'hundert ', 'Ich würde ', 'dh ', 'im ', 'sofortig ', 'sofort ', 'Bedeutung ', 'wichtig ', 'inc ', 'tatsächlich ', 'Index ', 'Information ', 'stattdessen ', 'Erfindung ', 'innere ', 'usw. ', 'es wird ', 'j ', 'zu ', 'behalten ', 'hält ', 'gehalten ', 'kg ', 'km ', 'kennt ', 'bekannt ', 'weiß ', 'l ', 'weitgehend ', 'zuletzt ', 'in letzter Zeit ', 'später ', 'letztere ', 'zuletzt ', 'am wenigsten ', 'weniger ', 'damit nicht ', 'Lassen ', 'Lasst uns ', 'mögen ', 'gefallen ', 'wahrscheinlich ', 'Linie ', 'wenig ', 'werde ', 'aussehen ', 'suchen ', 'sieht aus ', 'GmbH ', 'gemacht ', 'hauptsächlich ', 'machen ', 'macht ', 'viele ', 'kann ', 'könnte sein ', 'bedeuten ', 'meint ', 'inzwischen ', 'inzwischen ', 'nur ', 'mg ', 'könnte ', 'Million ', 'Fräulein ', 'ml ', 'Außerdem ', 'meist ', 'Herr', 'Herrin', 'Frau ', 'viel ', 'Becher ', 'Muss ', 'n ', 'auf ', 'Dann ', 'nämlich ', 'jetzt weiter ', 'nd ', 'in der Nähe von ', 'fast ', 'Notwendig ', 'notwendig ', 'brauchen ', 'Bedürfnisse ', 'weder ', 'noch nie ', 'Dennoch ', 'Neu ', 'Nächster ', 'neun ', 'neunzig ', 'niemand ', 'nicht ', 'keiner ', 'dennoch ', 'Niemand ', 'normalerweise ', 'wir ', 'notiert ', 'nichts ', 'nirgends ', 'erhalten ', 'erhalten ', 'offensichtlich ', 'häufig ', 'Oh ', 'in Ordnung ', 'in Ordnung ', 'alt ', 'weggelassen ', 'einer ', 'Einsen ', 'auf zu ', 'Wörter ', 'Andere ', 'Andernfalls ', 'draußen ', 'insgesamt ', 'geschuldet ', 'p ', 'Seite ', 'Seiten ', 'Teil ', 'besonders ', 'insbesondere ', 'Vergangenheit ', 'zum ', 'vielleicht ', 'platziert ', 'Bitte ', 'Mehr ', 'schlecht ', 'möglich ', 'möglicherweise ', 'möglicherweise ', 'pp ', 'überwiegend ', 'Geschenk ', 'vorher ', 'in erster Linie ', 'wahrscheinlich ', 'sofort ', 'stolz ', 'bietet ', 'stellen ', 'q ', 'Was ', 'schnell ', 'ziemlich ', 's ', 'r ', 'Na sicher ', 'lieber ', 'rd ', 'leicht ', 'Ja wirklich ', 'kürzlich ', 'vor kurzem ', 'ref ', 'refs ', 'hinsichtlich ', 'ungeachtet ', 'Grüße ', 'verbunden ', 'verhältnismäßig ', 'Forschung ', 'beziehungsweise ', 'resultierte ', 'resultierend ', 'Ergebnisse ', 'Recht ', 'Lauf ', 'sagte ', 'sah ', 'sagen ', 'Sprichwort ', 'sagt ', 'sek ', 'Sektion ', 'sehen ', 'Sehen ', 'scheinen ', 'schien ', 'scheinbar ', 'scheint ', 'gesehen ', 'selbst ', 'Selbst ', 'geschickt ', 'Sieben ', 'mehrere ', 'soll ', 'Schuppen ', 'shes ', 'Show ', 'gezeigt ', 'gezeigt ', 'gezeigt ', 'zeigt an ', 'von Bedeutung ', 'bedeutend ', 'ähnlich ', 'ähnlich ', 'schon seit ', 'sechs ', 'leicht ', 'jemanden ', 'irgendwie ', 'jemand ', 'etwas ', 'etwas ', 'irgendwann ', 'manchmal ', 'etwas ', 'irgendwo ', 'demnächst ', 'Es tut uns leid ', 'speziell ', 'spezifizierten ', 'angeben ', 'spezifizieren ', 'immer noch ', 'halt ', 'stark ', 'sub ', 'im Wesentlichen ', 'erfolgreich ', 'ausreichend ', 'vorschlagen ', 'sup ', 'sicher ', 'nehmen ', 'genommen ', 'nehmen ', 'sagen ', 'neigt dazu ', 'th ', 'danken ', 'Vielen Dank ', 'Danke ', 'das ist ', 'das haben ', 'von dort ', 'danach ', 'damit ', 'das Rote ', 'deshalb ', 'darin ', 'da wird ', 'davon ', 'da ', 'theres ', 'dazu ', 'daraufhin ', 'da haben ', 'Sie würden ', 'Sie sind ', 'Überlegen ', 'du ', 'obwohl ', 'obwohl ', 'tausend ', 'durch ', 'während ', 'durch ', 'so ', 'zu ', 'Trinkgeld ', 'zusammen ', 'dauerte ', 'zu ', 'gegenüber ', 'versucht ', 'versucht es ', 'wirklich ', 'Versuchen ', 'versuchen ', 'ts ', 'zweimal ', 'zwei ', 'u ', 'ein ', 'Unglücklicherweise ', 'es sei denn ', 'nicht wie ', 'unwahrscheinlich ', 'zu ', 'auf ', 'UPS ', 'uns ', 'verwenden ', 'gebraucht ', 'nützlich ', 'nützlich ', 'Nützlichkeit ', 'Verwendet ', 'mit ', 'meistens ', 'im ', 'Wert ', 'verschiedene ', "'und ", 'über ', 'nämlich ', 'vol ', 'Flüge ', 'vs. ', 'im ', 'wollen ', 'will ', 'war nicht ', 'Weg ', 'heiraten ', 'herzlich willkommen ', 'ging ', 'werent ', 'wie auch immer ', 'was wird ', 'was ist ', 'woher ', 'wann immer ', 'danach ', 'wohingegen ', 'wodurch ', 'worin ', 'wo ist ', 'worauf ', 'wo auch immer ', 'ob ', 'Laune ', 'wohin ', 'Wer würde ', 'wer auch immer ', 'ganze ', 'Wer wird ', 'wen auch immer ', 'wer ', 'deren ', 'weit ', 'bereit ', 'Wunsch ', 'innerhalb ', 'ohne ', 'Gewohnheit ', 'Wörter ', 'Welt ', 'würde nicht ', 'www ', 'x ', 'Ja ', 'noch ', 'du ', 'du bist ', 'mit ', 'Null ', 'wie ', 'ist nicht ', 'ermöglichen ', 'erlaubt ', 'ein Teil ', 'erscheinen ', 'schätzen ', 'angemessen ', 'damit verbundenen ', 'Beste ', 'besser ', 'Komm schon ', "c's ", 'kippen ', 'Änderungen ', 'deutlich ', 'über ', 'Folglich ', 'Erwägen ', 'in Anbetracht ', 'dazugehörigen ', 'Kurs ', 'zur Zeit ', 'bestimmt ', 'beschrieben ', 'Trotz ', 'vollständig ', 'genau ', 'Beispiel ', 'gehen ', 'Schöne Grüße ', 'Hallo ', 'Hilfe ', 'hoffnungsvoll ', 'ignoriert ', 'insofern ', 'zeigen ', 'angegeben ', 'zeigt an ', 'innere ', 'soweit ', 'es würde ', 'behalten ', 'hält ', 'Roman ', 'vermutlich ', 'vernünftig ', 'zweite ', 'zweitens ', 'sinnvoll ', 'ernst ', 'Ernsthaft ', 'sicher ', "t's ", 'dritte ', 'gründlich ', 'gründlich ', 'drei ', 'Gut ', 'Wunder ', 'ein ', 'Über ', 'über ', 'über ', 'über ', 'nach ', 'danach ', 'nochmal ', 'gegen ', 'alle ', 'fast ', 'allein ', 'entlang ', 'bereits ', 'ebenfalls ', 'obwohl ', 'immer ', 'bin ', 'unter ', 'unter ', 'unter ', 'Menge ', 'ein ', 'und ', 'Ein weiterer ', 'irgendein ', 'jedenfalls ', 'jemand ', 'etwas ', 'wie auch immer ', 'irgendwo ', 'sind ', 'um ', 'wie ', 'beim ', 'zurück ', 'Sein ', 'wurden ', 'weil ', 'werden ', 'wird ', 'Werden ', 'gewesen ', 'Vor ', 'vorweg ', 'hinter ', 'Sein ', 'unten ', 'neben ', 'Außerdem ', 'zwischen ', 'darüber hinaus ', 'Rechnung ', 'beide ', 'Unterseite ', 'aber ', 'durch ', 'Anruf ', 'können ', 'kann nicht ', 'kippen ', 'Was ', 'mit ', 'könnten ', 'konnte nicht ', 'Schrei ', 'von ', 'beschreiben ', 'Detail ', 'tun ', 'getan ', 'Nieder ', 'fällig ', 'während ', 'jeder ', 'z.B ', 'acht ', 'entweder ', 'elf ', 'sonst ', 'anderswo ', 'leer ', 'genug ', 'usw ', 'sogar ', 'je ', 'jeder ', 'jeder ', 'alles ', 'überall ', 'außer ', 'wenige ', 'fünfzehn ', 'fify ', 'füllen ', 'finden ', 'Feuer ', 'zuerst ', 'fünf ', 'zum ', 'ehemalige ', 'früher ', 'vierzig ', 'gefunden ', 'vier ', 'von ', 'Vorderseite ', 'voll ', 'des Weiteren ', 'bekommen ', 'geben ', 'gehen ', 'hätten ', 'hat ', 'hat nicht ', 'haben ', 'mit ', 'daher ', 'ihm', 'ihr', 'Hier ', 'Jenseits ', 'hiermit ', 'hierin ', 'hierauf ', 'ihres ', 'Sie selber ', 'ihm ', 'selbst ', 'seine ', 'Wie ', 'jedoch ', 'hundert ', 'dh ', 'wenn ', 'im ', 'inc ', 'tatsächlich ', 'Interesse ', 'in ', 'ist ', 'es ', 'es ist ', 'selbst ', 'behalten ', 'zuletzt ', 'letztere ', 'zuletzt ', 'am wenigsten ', 'weniger ', 'GmbH ', 'gemacht ', 'viele ', 'kann ', 'mich ', 'inzwischen ', 'könnte ', 'von ', 'Bergwerk ', 'Mehr ', 'Außerdem ', 'die meisten ', 'meist ', 'Bewegung ', 'viel ', 'Muss ', 'meine ', 'mich selber ', 'Dann ', 'nämlich ', 'weder ', 'noch nie ', 'Dennoch ', 'Nächster ', 'neun ', 'Nein ', 'niemand ', 'keiner ', 'Niemand ', 'Noch ', 'nicht ', 'nichts ', 'jetzt ', 'nirgends ', 'von ', 'aus ', 'häufig ', 'auf ', 'Einmal ', 'einer ', 'nur ', 'auf zu ', 'oder ', 'andere ', 'Andere ', 'Andernfalls ', 'unsere ', 'unsere ', 'uns selbst ', 'aus ', 'Über ', 'besitzen ', 'Teil ', 'zum ', 'vielleicht ', 'Bitte ', 'stellen ', 'lieber ', 'Hitze ', 'gleich ', 'sehen ', 'scheinen ', 'schien ', 'scheinbar ', 'scheint ', 'ernst ', 'mehrere ', 'Schlange ', 'sollte ', 'Show ', 'Seite ', 'schon seit ', 'aufrichtig ', 'sechs ', 'sechzig ', 'damit ', 'etwas ', 'irgendwie ', 'jemand ', 'etwas ', 'irgendwann ', 'manchmal ', 'irgendwo ', 'immer noch ', 'eine solche ', 'System ', 'nehmen ', 'diese ', 'als ', 'Das ', 'das ', 'ihr ', 'Sie ', 'sich ', 'dann ', 'von dort ', 'Dort ', 'danach ', 'damit ', 'deshalb ', 'darin ', 'daraufhin ', 'diese ', 'Sie ', 'dick ', 'dünn ', 'dritte ', 'diese ', 'jene ', 'obwohl ', 'drei ', 'durch ', 'während ', 'durch ', 'so ', 'zu ', 'zusammen ', 'auch ', 'oben ', 'zu ', 'gegenüber ', 'zwölf ', 'zwanzig ', 'zwei ', 'ein ', 'unter ', 'bis um ', 'oben ', 'auf ', 'uns ', 'sehr ', 'über ', 'war ', 'wir ', 'Gut ', 'wurden ', 'Was ', 'wie auch immer ', 'wann ', 'woher ', 'wann immer ', 'wo ', 'danach ', 'wohingegen ', 'wodurch ', 'worin ', 'worauf ', 'wo auch immer ', 'ob ', 'welche ', 'während ', 'wohin ', 'Wer ', 'wer auch immer ', 'ganze ', 'wem ', 'deren ', 'Warum ', 'werden ', 'mit ', 'innerhalb ', 'ohne ', 'würde ', 'noch ', 'Haben ', 'Ihre ', 'deine ', 'du selber ', 'euch ', 'das ', 'Mangel ', 'machen ', 'wollen ', 'scheinen ', 'Lauf ', 'brauchen ', 'sogar ', 'Recht ', 'verwenden ', 'nicht ', 'würde ', 'sagen ', 'könnten ', '_ ', 'Sein ', 'kennt ', 'gut ', 'gehen ', 'bekommen ', 'tun ', 'getan ', 'Versuchen ', 'viele ', 'von ', 'Gegenstand ', 'Hitze ', 'Quote ', 'etwas ', 'nett ', 'danken ', 'Überlegen ', 'sehen ', 'lieber ', 'einfach ', 'leicht ', 'Menge ', 'Linie ', 'sogar ', 'ebenfalls ', 'kann ', 'nehmen ', 'Kommen Sie '])
a_file = open("stopwords.txt", "r")
for line in a_file:
    stripped_line = line.strip()
    stop_words_de.append(stripped_line)


def top_words(mgp, cluster_word_distribution, top_cluster, values):
    """funtion to extract the top topic keywords and their occurences return it as dictionary
    in: mgp : the model GSDMM
    in: cluster_word_distribution : dictionary containing all topic keywords and their occurences
    in: top_cluster: number of cluster requested 
    in: values: number of topic keywords requested
    out: dictionary that contains topics and keywords
    """
    dictionary = {}
    for cluster in top_cluster:
      sort_dicts =sorted(mgp.cluster_word_distribution[cluster].items(), key=lambda k: k[1], reverse=True)[:values]
        
      st.write('Cluster %s : %s'%(cluster,sort_dicts))
      dictionary[cluster] = sort_dicts
        
    return dictionary


def top_docs(mgp, cluster_doc_frequency, top_cluster, values):
    """funtion to extract the top topic keywords and their document frequencies return it as dictionary
    in: mgp : the model GSDMM
    in: cluster_doc_frequency : dictionary containing all topic keywords and their document frequencies
    in: top_cluster: number of cluster requested 
    in: values: number of topic keywords requested
    out: dictionary that contains topics and keywords
    """
    dictionary = {}
    for cluster in top_cluster:
      sort_dicts =sorted(mgp.cluster_doc_frequency[cluster].items(), key=lambda k: k[1], reverse=True)[:values]

      st.write('Cluster %s : %s'%(cluster,sort_dicts))
            
      dictionary[cluster] = sort_dicts
        
    return dictionary


def top_words_docs(mgp, cluster_word_doc_frequency, top_cluster, values):
    """funtion to extract the top topic keywords and their document frequencies times log of 10(occurences) return it as dictionary
    in: mgp : the model GSDMM
    in: cluster_word_doc_frequency : dictionary containing all topic keywords and their document frequencies times log of 10(occurences)
    in: top_cluster: number of cluster requested 
    in: values: number of topic keywords requested
    out: dictionary that contains topics and keywords
    """
    dictionary = {}
    for cluster in top_cluster:
      sort_dicts =sorted(mgp.cluster_word_doc_frequency[cluster].items(), key=lambda k: k[1], reverse=True)[:values]
      sort_dicts = [words for words in sort_dicts  if words[1] != 0]
         
      st.write('Cluster %s : %s'%(cluster,sort_dicts))
            
      dictionary[cluster] = sort_dicts
        
    return dictionary


def get_model_results_gsdmm(df, deleted_index, n_of_words,texts,mgp=None):
    """funtion to extract model result of LEVEL 1 such as topics, keywords and insert in a pandas dataframe
    in: df : pd.DataFrame() that contains the data input text
    in: n_of_words : number of topics keywords requested
    in: mgp: the trained model
    in: deleted_index: list of indexes where the text was deleted
    in: texts: list of list of words
    out: dataframe and dictionary that contains topics and keywords
    """

    #INITIALIZING
    words_list = []
    doc_number_list = []

    #Show the number of documents per topic
    doc_count = np.array(mgp.cluster_doc_count)
    st.write('Number of documents per topic (LEVEL 1) :', doc_count)

    # Topics sorted by the number of document they are allocated to
    top_index = doc_count.argsort()[-30:][::-1]
    
    # Show the document frequencies dictionary
    st.success('Dictionary for doc frequency (LEVEL 1)')
    dictionary_docs = top_docs(mgp, mgp.cluster_doc_frequency, top_index, n_of_words)

    # Show the document frequencies times log10(occurences) dictionary
    #st.success('Dictionary for word document frequency')
    #dictionary_words_docs = top_words_docs(mgp, mgp.cluster_word_doc_frequency, top_index, n_of_words)

    # Show the keywords and their occurences dictionary
    #st.success('Dictionary for top keywords (LEVEL 1)')    
    #dictionary = top_words(mgp, mgp.cluster_word_distribution, top_index, n_of_words)

    #Extracting topic keywords of each text   
    for i in range(0,len(texts)):
        words = []
        doc_number_list.append(int(doc_count[mgp.choose_best_label(texts[i])[0]]))
        
        for j in range(0,n_of_words):
            try:
                words.append(dictionary_docs[mgp.choose_best_label(texts[i])[0]][j][0])                
            except:
                pass        
        words_list.append(words)
    
    #Inserting empty space in the list for every text deleted
    for i in range(len(deleted_index)):
        if deleted_index !=[]:
            doc_number_list.insert(deleted_index[i], '')
            words_list.insert(deleted_index[i], '')    

    # Inserting into the dataframe topic keywords and number of documents per topic columns
    df.insert(loc=3, column='no_of_docs_per_topic (LEVEL 1)', value=doc_number_list)
    df.insert(loc=4, column='topic_keywords (LEVEL 1)', value=words_list)

    # Extracting the topic number of each text and inserting it into the dataframe
    topic_number = [mgp.choose_best_label(texts[i])[0] for i in range(0,len(texts))]
    for i in range(len(deleted_index)):
        if deleted_index !=[]:
            topic_number.insert(deleted_index[i], '')        
    df.insert(loc=3, column='Topic number', value=topic_number)
    
    
    return df, dictionary_docs


def generate_labels_de(dictionary,deleted_index, topics_df, n_of_topics):
    """funtion to generate labels for each document list LEVEL 1
    in: dictionary : dictionary that contains topic keywords
    in: n_of_topics : number of topics requested
    in: topics_df: dataframe that contains text and the topic number of each text
    in: deleted_index: list of indexes where the text was deleted    
    out: list of labels and document list of each topic for LEVEL 2
    """

    # INILIZING THE LISTS
    labels_count = ['None'] * n_of_topics
    adj_noun = [[] for i in range(n_of_topics)]
    adj_noun_sorted = [[] for i in range(n_of_topics)]
    topics_document_list = []
    topic_label = []
    
    #GOING THROUGH EACH TOPIC EXTRACT DOCUMENT LIST
    for i in range(0, n_of_topics):
        document_list = topics_df[topics_df['Topic number'] == i]['Text'].tolist()        
       
        if document_list != []:            
            topics_document_list.append(document_list)
            
            #Extracting the topic keywords into list
            labels = []
            for l in range(len(dictionary[i])):
                labels.append(dictionary[i][l][0])               
                
        
            # Extracting most frequent Bigrams from document list
            vectorizer = CountVectorizer( ngram_range=(2,2), tokenizer=None, preprocessor=None, lowercase=False )
            vectors = vectorizer.fit_transform(document_list)
            feature_names = vectorizer.get_feature_names()
            dense = vectors.todense()
            denselist = dense.tolist()
            df_count = pd.DataFrame(denselist, columns=feature_names)        
            max_count = df_count.max()
            sorted = max_count.sort_values(ascending=False)

            #Extracting most frequent bigrams that are ADJ + NOUN and EITHER ADJ OR NOUN MUST be a topic keyword
            for j in range(0,len(sorted)):
                pos_tag = []
                label_list = []            
                bigram = nltk.tokenize.word_tokenize(sorted.index[j],language='german')
                tags = tagger.tag_sent(bigram)
                for (word,lemma,pos) in tags:
                    label_list.append(lemma)
                    pos_tag.append(pos)

                if pos_tag == ['ADJA','NN'] and 'anderer' not in label_list:
                    for word in label_list:
                        if word in labels and label_list[0] + ' ' + label_list[1]  not in adj_noun[i]:
                            adj_noun[i].append(label_list[0] + ' ' + label_list[1])

            # SORT them out according to the order of the dictionary of topic keywords
            for l in range(len(dictionary[i])):
                for k in range(len(adj_noun[i])):
                    if dictionary[i][l][0] in adj_noun[i][k]:
                        adj_noun_sorted[i].append(adj_noun[i][k])                 
            

            if adj_noun_sorted[i] != []:
                labels_count[i] = adj_noun_sorted[i][0]         

                  
    st.write(labels_count)
    #st.success('ALL ADJ + NOUNS GENERATED FOR EACH TOPIC')
    #st.write(adj_noun_sorted)   

    # Extracting the topic label of each text and inserting it in the dataframe
    for i in range(len(topics_df['Topic number'])):
        if i not in deleted_index:
          topic_label.append(labels_count[int(topics_df['Topic number'][i])])
        else:
          topic_label.append('')

    topics_df.insert(loc=4, column='topic_label (LEVEL 1)', value=topic_label)


    return labels_count, topics_document_list  


def generate_summary_de(topics_df,deleted_index):
    """funtion to generate a list of 10 document keywords for each document    
    in: topics_df: dataframe that contains text and the topic number of each text
    in: deleted_index: list of indexes where the text was deleted    
    out: topics_df: dataframe with the list of document keywords
    """

    #INITIALIZING LISTS
    summary_count_list = []
    summary_words_count = []
        
    
    for i in range(0,len(topics_df['Text'])):
        if i not in deleted_index:       
        
          document = topics_df['Text'][i]                   
          summary_count = []
          summary_words = []

          #EXTRACTING most frequent nouns and adjs from each document
          vectorizer = CountVectorizer(ngram_range=(1,1),tokenizer=None, preprocessor=None, lowercase=False, stop_words = stop_words_de)
          vectors = vectorizer.fit_transform([document])
          feature_names = vectorizer.get_feature_names()
          dense = vectors.todense()
          denselist = dense.tolist()
          df_tfidf = pd.DataFrame(denselist, columns=feature_names)
          max = df_tfidf.max()
          sorted = max.sort_values(ascending=False)
          for j in range(0,len(sorted)):
              tokenized = nltk.tokenize.word_tokenize(sorted.index[j],language='german')
              tags = tagger.tag_sent(tokenized)
              
              for (word,lemma,pos)  in tags:
                  
                  if pos == "NN" or pos == "ADJA":               
              
                    summary_words.append(lemma)
                    summary = '(' + str(lemma) + ', ' + str(sorted[j]) + ' )'
                    summary_count.append(summary)

              if len(summary_count) == 10:
                break

          summary_count_list.append(summary_count)
          summary_words_count.append(summary_words)

        else:
          summary_count = []
          summary_words = []
          summary_words_count.append(summary_words)
          summary_count_list.append(summary_count)

    
    topics_df.insert(loc=7, column='document_keywords', value=summary_count_list)


    return topics_df  


