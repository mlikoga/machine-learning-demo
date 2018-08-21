#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import division
from sklearn import tree
from collections import Counter
import re

# Definiçao das características e da saída
feature_names = ['K','W','Y'] 
classes_names = { 0: 'English', 1: 'Portugues' }

# 2. Extrair as características
def extract_features(phrase):
  phrase_only_letters = re.sub(r"[^A-Z]", "", phrase.upper())
  letter_freq = Counter(phrase_only_letters) # {'A': 5, 'B': 1, 'C': 2, 'D': 1}
  example_features = []
  for letter in feature_names:
    example_features.append(letter_freq[letter] / sum(letter_freq.values()))
  
  return example_features

features = []

features.append(extract_features("Where in the world is she?"))
features.append(extract_features("May the Force be with you."))
features.append(extract_features("I find your lack of faith disturbing."))

features.append(extract_features("Que a força esteja com voce"))
features.append(extract_features("Onde no mundo está ela?"))
features.append(extract_features("Sua falta de fé é perturbadora"))

classes = [0, 0, 0, 1, 1, 1]

# 3. Treinar
classifier = tree.DecisionTreeClassifier()
classifier = classifier.fit(features, classes)

# 4. Classificar novos dados
inputStr = raw_input("\nEscreva sua frase aqui: ")
while inputStr:
  input = extract_features(inputStr)
  output = classifier.predict([input])
  print "Essa entrada foi classificada como: " + classes_names[output[0]]
  inputStr = raw_input("\nEscreva sua frase aqui: ")
