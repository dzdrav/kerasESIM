import xmltodict
import json
import html
from bs4 import BeautifulSoup
from bs4.diagnose import diagnose

"""
dodaje svakom elementu liste 'names' sufiks 'suffix' odvojeno sa 'separator'
primjer:
  names = [input, output]
  suffix = 'mini'
  return [input_mini, output_mini]
služi za korištenje manjih datasetova (pri testiranju)
"""
def addSufixToStrings(names, suffix, separator = '_'):
    completeNameList, resultList = [], []
    for name in names:
        completeNameList.append(name.split('.'))
        for completeName in completeNameList:
            resultList.append(completeName[0] + separator + suffix + '.' + completeName[1])
            return resultList

"""
skup obrada koje želimo nad rečenicom parsiranom iz XML-a
escapeanje HTML elemenata, uklanjanje praznina, line breakova itd.
vraća normaliziran string
"""
def normalizeString(sentence):
    sentence = sentence.strip()
    sentence = sentence.replace("\n", "")
    sentence = html.unescape(sentence)
    return sentence

"""
parsira originalni XML RTE dataseta u dvije TXT datoteke: train i test set
one služe kao ulaz importiranim funkcijama za preprocesiranje (dolje)
"""
def XML_to_TXT(filenameIn, filenameOutTrain, filenameOutTest, mini = False):
    if mini:
        filenameIn, filenameOutTrain, filenameOutTest = addSufixToStrings([filenameIn, filenameOutTrain, filenameOutTest], 'mini')
    print("Input: " + filenameIn, "Output: " + filenameOutTrain + ", " + filenameOutTest)
    allRows = []
    with open(filenameIn, 'r', encoding='utf-8') as inFile:
        # pročitaj cijelu datoteku
        completeFile = inFile.read()
        # parsiraj sve pročitano
        soup = BeautifulSoup(completeFile, 'html.parser')
        # jedan par je cijeli HTML element između <pair></pair> tagova
        allPairs = soup.find_all('pair')
        allRows = []
        for pair in allPairs:
            allRows.append('sentence1 ' + normalizeString(pair.t.string))
            allRows.append('sentence1 ' + normalizeString(pair.h.string))
            allRows.append({'TRUE': 'Y', 'FALSE': 'N'}[pair['value']])
    with open(filenameOutTrain, 'w', encoding='utf-8') as outTrain, open(filenameOutTest, 'w', encoding='utf-8') as outTest:
        border = int((len(allRows) // 3) * 0.9) * 3
        print("Train pairs: ", int(border / 3))
        print("Test pairs: ", int((len(allRows) - border) / 3))
        for index, row in enumerate(allRows):
            if index < border:
                outTrain.write("%s\n" % row)
            else:
                outTest.write("%s\n" % row)

if  __name__ == '__main__':
    """
    preprocess RTE
    """
    input = "dataset.xml"
    output_train = "RTE/RTE_train.txt"
    output_test = "RTE/RTE_test.txt"
    XML_to_TXT(input, output_train, output_test, mini = False)
