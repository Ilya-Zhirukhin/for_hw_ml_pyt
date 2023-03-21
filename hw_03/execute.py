

from dataset.sportsmans_height import Sportsmanheight
from model.simple_classifier import Classifier
import numpy as np
import pandas as pd
from config.cfg import cfg
import plotly.graph_objects as go
import copy

# Step 2: Get dataset and model predictions
dataset = Sportsmanheight()()
predictions = Classifier()(dataset['height'])

# Step 3:
# Compute TP, FP, FN, TN, accuracy, recall, precision, and F1 score for different thresholds


thresholds = np.unique(predictions)
results = []
for threshold in thresholds:
    preds = (predictions >= threshold).astype(int)
    TP = np.sum(np.logical_and(preds == 1, dataset['class'] == 1))
    FP = np.sum(np.logical_and(preds == 1, dataset['class'] == 0))
    FN = np.sum(np.logical_and(preds == 0, dataset['class'] == 1))
    TN = np.sum(np.logical_and(preds == 0, dataset['class'] == 0))
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    recall = TP / (TP + FN)
    precision = TP / (TP + FP)
    if precision + recall == 0:
        f1_score = 0
    else:
        f1_score = 2 * precision * recall / (precision + recall)
    if precision + recall == 0:
        f1_score = 0
    else:
        f1_score = 2 * precision * recall / (precision + recall)

    results.append([threshold, TP, FP, FN, TN, accuracy, recall, precision, f1_score])

# Step 4
results = pd.DataFrame(results, columns=['threshold', 'TP', 'FP', 'FN', 'TN', 'accuracy', 'recall', 'precision', 'f1_score'])
results = results.sort_values(by='threshold', ascending=False)
fig = go.Figure()
fig.add_trace(go.Scatter(x=results['recall'], y=results['precision'], mode='lines'))
fig.update_layout(title='Precision-Recall Curve (AUC={:.4f})'.format(results['accuracy'].mean()),
                  xaxis_title='Recall', yaxis_title='Precision')
fig.update_traces(hovertemplate='Threshold=%{text:.2f}<br>Precision=%{y:.4f}<br>Recall=%{x:.4f}<br>Accuracy=%{customdata[0]:.4f}<br>F1 Score=%{customdata[1]:.4f}')
fig.update_traces(text=results['threshold'], customdata=[results['accuracy'], results['f1_score']])
fig.show()

# Step 5
TPRs = results['recall']
FPRs = results['FP'] / (results['FP'] + results['TN'])
fig = go.Figure()
fig.add_trace(go.Scatter(x=FPRs, y=TPRs, mode='lines'))
fig.update_layout(title='ROC Curve (AUC={:.4f})'.format(np.trapz(TPRs, x=FPRs)),
                  xaxis_title='False Positive Rate', yaxis_title='True Positive Rate')
fig.update_traces(hovertemplate='Threshold=%{text:.2f}<br>TPR=%{y:.4f}<br>FPR=%{x:.4f}<br>Accuracy=%{customdata[0]:.4f}<br>F1 Score=%{customdata[1]:.4f}')
fig.update_traces(text=results['threshold'], customdata=[results['accuracy'], results['f1_score']])
fig.show()
