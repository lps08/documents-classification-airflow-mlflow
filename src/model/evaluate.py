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
        """
        Calculate various evaluation metrics for the model's performance and return them as a dictionary.

        Returns:
        dict: A dictionary containing the following evaluation metrics:
            - 'cross validation': List of cross-validation scores.
            - 'cross validation mean': Mean cross-validation score.
            - 'recall': List of recall scores for each class.
            - 'recall mean': Mean recall score.
            - 'F1 score': List of F1 scores for each class.
            - 'F1 score mean': Mean F1 score.
            - 'R2 score': R-squared score.
            - 'Precision score': List of precision scores for each class.
            - 'Precision score mean': Mean precision score.
            - 'MCC score': Matthews correlation coefficient score.

        Example:
        ```
        evaluation_metrics = score_model()
        print("Cross Validation Scores:", evaluation_metrics['cross validation'])
        print("Mean Cross Validation Score:", evaluation_metrics['cross validation mean'])
        # Print other evaluation metrics as needed...
        ```
        This method calculates and returns various evaluation metrics for the model's performance, including cross-validation,
        recall, F1 score, R-squared score, precision, and Matthews correlation coefficient (MCC).
        """
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
        """
        Plot a confusion matrix for the model's predictions on the test data.

        Parameters:
        name (str, optional): A name or title for the confusion matrix plot. Default is None.

        Returns:
        matplotlib.figure.Figure: The figure containing the confusion matrix plot.

        Example:
        ```
        # Assuming 'evaluation' is an instance of the class with this method
        fig = evaluation.plot_confusion_matrix(name="Confusion Matrix")
        plt.show()
        ```
        This method generates and displays a confusion matrix plot for the model's predictions on the test data.
        You can provide a name or title for the plot, which is optional.
        """
        """Plot confusion matrix"""
        cf_matrix = confusion_matrix(y_true=self.y_test, y_pred=self.pred)
        
        _, ax = plt.subplots(figsize=(15,10))         # Sample figsize in inches
        heapmap = sns.heatmap(cf_matrix, annot=True, cmap='Blues', fmt="d", linewidths=.5, ax=ax)

        return heapmap.get_figure()

    def __cross_validation_score(self):
        """
        Calculate the cross-validation accuracy score for the model.

        Returns:
        numpy.ndarray: An array containing cross-validation accuracy scores for each fold.

        Example:
        ```
        # Assuming 'evaluation' is an instance of the class with this method
        cv_scores = evaluation._Evaluation__cross_validation_score()
        print("Cross-Validation Scores:", cv_scores)
        ```
        This method performs k-fold cross-validation (with k=5 by default) and returns an array of accuracy scores
        for each fold. It provides an estimate of the model's performance across different subsets of the training data.
        """
        """Get the cross validation score"""
        kfold = KFold(n_splits=5)
        return cross_val_score(self.model, self.X_train, self.y_train, cv=kfold, scoring='accuracy')     
