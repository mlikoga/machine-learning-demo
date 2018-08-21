from sklearn import tree

feature_names = ['Calibre', 'Comprimento']
features = [[10,20], [5,7], [13,15], [3,10], [15,20], [16,23], [13,19], [14,17]]
class_names = ['A', 'B']
labels = [0, 0, 0, 0, 1, 1, 1, 1]

classifier = tree.DecisionTreeClassifier()
classifier = classifier.fit(features, labels)


# Print tree in PDF
import graphviz 
dot_data = tree.export_graphviz(classifier, out_file=None, 
                     feature_names=feature_names,
                     class_names=class_names,
                     filled=True, rounded=True,
                     impurity=False)
graph = graphviz.Source(dot_data) 
graph.render("tree")