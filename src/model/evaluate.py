from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import confusion_matrix, recall_score, f1_score, r2_score, precision_score, matthews_corrcoef
import matplotlib.pyplot as plt
import seaborn as sns

class Evaluate():
    def __init__(self, model, X_train, y_train, X_test, y_test):
        self.model = model
        self.X_train = X_train
        self.y_train = y_train
        self.y_test = y_test
        
        #storage all the results
        self.results = {}

        self.pred = self.model.predict(X_test)

    def score_model(self):
        cross_score = self.__cross_validation_score()
        recall = recall_score(self.y_test, self.pred, average=None)
        f1 = f1_score(self.y_test, self.pred, average=None)
        r2 = r2_score(self.y_test, self.pred)
        precision = precision_score(self.y_test, self.pred, average=None)
        mcc = matthews_corrcoef(self.y_test, self.pred)   

        return {'cross validation': cross_score,
                'cross validation mean': sum(cross_score)/len(cross_score),
                'recall' : recall,
                'recall mean': sum(recall)/len(recall),
                'F1 score': f1,
                'F1 score mean': sum(f1)/len(f1),
                'R2 score': r2,
                'Precision score': precision,
                'Precision score mean': sum(precision)/len(precision),
                'MCC score': mcc}

    def plot_confusion_matrix(self, name:str=None):
        """Plot confusion matrix"""
        cf_matrix = confusion_matrix(y_true=self.y_test, y_pred=self.pred)
        
        _, ax = plt.subplots(figsize=(15,10))         # Sample figsize in inches
        heapmap = sns.heatmap(cf_matrix, annot=True, cmap='Blues', fmt="d", linewidths=.5, ax=ax)

        return heapmap.get_figure()

    def __cross_validation_score(self):
        """Get the cross validation score"""
        kfold = KFold(n_splits=5)
        return cross_val_score(self.model, self.X_train, self.y_train, cv=kfold, scoring='accuracy')     
