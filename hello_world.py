#!/usr/bin/env python
# -*- coding: utf-8 -*-
from sklearn import tree

# 2. Extrair as features
feature_names = ['Calibre', 'Comprimento']
features = [[10,20], [5,7], [13,15], [3,10], [15,20], [16,23], [13,19], [14,17]]
classes = {0: 'A', 1: 'B'}
labels = [0, 0, 0, 0, 1, 1, 1, 1]

# 3. Treinar
classifier = tree.DecisionTreeClassifier()
classifier = classifier.fit(features, labels)

# Imprimir PDF com árvore
import graphviz 
dot_data = tree.export_graphviz(classifier, out_file=None, 
                     feature_names=feature_names,
                     class_names=classes.values(),
                     filled=True, rounded=True,
                     impurity=False)
graph = graphviz.Source(dot_data) 
graph.render("tree")

# 4. Classificar novos dados
while True:
  inputStr = raw_input("\nInsira novo dado de entrada: ")
  input = map(int, inputStr.split(','))
  print "Essa entrada foi classificada como: " + classes[classifier.predict([input])[0]]