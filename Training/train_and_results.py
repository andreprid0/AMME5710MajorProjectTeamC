import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

#train parameters--------------------
k_values = list(range(1,16,2))
kernels = ["linear", "rbf", "poly"]
Cs = [0.1, 0.3, 1, 2, 5, 10]
#--------------------------------------

def train_knn(train_images, train_labels, k_values=k_values):
    models = {}
    for k in k_values:
        clf = KNeighborsClassifier(n_neighbors=k)
        clf.fit(train_images, train_labels)
        models[k]=clf
    return models 

def train_svm(train_images, train_labels, kernels=kernels, Cs=Cs):
    models = {}
    for kernel in kernels: 
        for C in Cs: 
            if kernel == "linear":
                #Train SVM with kernel "linear"
                clf = make_pipeline(StandardScaler(), SVC(kernel='linear', C=C)) #uses StandardScaler (normalisation)
                clf.fit(train_images, train_labels)

            elif kernel == "rbf":
                #Train SVM with kernel "rbf"
                clf = make_pipeline(StandardScaler(), SVC(kernel='rbf', C=C, gamma = "scale")) 
                clf.fit(train_images, train_labels)

            elif kernel == "poly":
                        #Train SVM with kernel "poly"
                clf = make_pipeline(StandardScaler(), SVC(kernel='poly', C=C, degree = 2, gamma = "scale")) 
                clf.fit(train_images, train_labels)

            models[(kernel,C)] = clf
    return models

def get_results_knn(models, test_images, test_labels):
    accuracies = {}
    best_k = None
    best_acc = -1.0
    best_metrics = None

    for k, clf in models.items(): 
        label_pred = clf.predict(test_images)
        acc = accuracy_score(test_labels, label_pred)
        accuracies[k] = acc

        if acc > best_acc:
            best_acc = acc
            best_k = k
            cm = confusion_matrix(test_labels, label_pred)
            report = classification_report(test_labels, label_pred, digits=3, zero_division=0)
            best_metrics = {
                "label_pred": label_pred,
                "confusion_matrix": cm,
                "classification_report": report,
            }

    return {
        "accuracies": accuracies,
        "best_k": best_k,
        "best_accuracy": best_acc,
        "best_metrics": best_metrics,
    }

def get_results_svm(models, test_images, test_labels): 
    accuracies = {}
    best_k = None
    best_acc = -1.0
    best_metrics = None

    for (kernel, C), clf in models.items():
        label_pred = clf.predict(test_images)
        acc = accuracy_score(test_labels, label_pred)
        accuracies[(kernel, C)] = acc
        if acc > best_acc:
            best_acc = acc
            best_key = (kernel, C)
            cm = confusion_matrix(test_labels, label_pred)
            report = classification_report(test_labels, label_pred, digits=3, zero_division=0)
            best_metrics = {
                "label_pred": label_pred,
                "confusion_matrix": cm,
                "classification_report": report,
            }

    return {
        "accuracies": accuracies,
        "best_params": {"kernel": best_key[0], "C": best_key[1]} if best_key else None,
        "best_accuracy": best_acc,
        "best_metrics": best_metrics,
    }