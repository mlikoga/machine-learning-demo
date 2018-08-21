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
  letter_freq = Counter(phrase_only_letters) # Counter({'A': 5, 'B': 1, 'C': 2, 'D': 1})
  example_features = []
  for letter in feature_names:
    example_features.append(letter_freq[letter] / sum(letter_freq.values()))
  
  return example_features

features = []
classes = []

english_phrases = filter(None, 
  """In 1984 Los Angeles, a cyborg assassin known as a Terminator arrives from 2029 and steals guns and clothes. 
  Shortly afterward, Kyle Reese, a human soldier from 2029, arrives. 
  He steals clothes and evades the police.
  The Terminator begins systematically killing women named Sarah Connor, whose addresses it finds in the telephone directory.
  It tracks the last Sarah Connor to a nightclub, but Kyle rescues her. 
  The pair steal a car and escape with the Terminator pursuing them in a police car.""".split("."))

portuguese_phrases = filter(None, 
  """A beira da extinçao, um homem chamado John Connor ira liderar os humanos num levante contra as maquinas.
  Ensinando formas de engana-las, John vai levar os humanos para uma vantagem. 
  A beira da destruiçao a skynet ira enviar um ciborgue Cyberdyne 101 para matar Sarah Connor, mae de John, antes mesmo dele nascer. 
  Apos capturarem um complexo de laboratorios, a resistencia descobre o plano da skynet e o tenente Kyle Reese se oferece para ir ao passado (no ano de 1984) proteger Sarah.
  Mesmo sabendo que o complexo sera destruido e ele e o exterminador ficarao presos no passado para sempre.
  Apos chegar ao presente o exterminador rouba uma loja de armas e Procura pelo endereço de Sarah Connor em uma lista telefonica e encontra tres mulheres com esse nome e vai ate elas matando duas delas restando apenas a Sarah certa.""".split("."))


all_phrases = english_phrases + portuguese_phrases
for phrase in all_phrases:
  features.append(extract_features(phrase))

for phrase in english_phrases:
  classes.append(0)
for phrase in portuguese_phrases:
  classes.append(1)

# 3. Treinar
classifier = tree.DecisionTreeClassifier()
classifier = classifier.fit(features, classes)

# Imprimir PDF com arvore
import graphviz 
dot_data = tree.export_graphviz(classifier, out_file=None, 
                     feature_names=feature_names,
                     class_names=classes_names.values(),
                     filled=True, rounded=True,
                     impurity=False)
graph = graphviz.Source(dot_data) 
graph.render("tree")

# 4. Classificar novos dados
while True:
  inputStr = raw_input("\nEscreva sua frase aqui: ")
  input = extract_features(inputStr)
  output = classifier.predict([input])[0]
  print "Essa entrada foi classificada como: " + classes_names[output]




