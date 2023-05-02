
from dataset_titanik import Titanic
from adaboost_students import Adaboost
import numpy as np
from sklearn.metrics import confusion_matrix


titanic = Titanic('titanik_train_data.csv', 'titanik_test_data.csv')
data = titanic()
train_data = data['train_input']
train_target = data['train_target']
test_data = data['test_input']
test_target = data['test_target']

adaboost = Adaboost(100)
adaboost.train(train_target, train_data)
test_prediction = adaboost.get_prediction(test_data)

conf_matrix = confusion_matrix(test_target, test_prediction)
precision = conf_matrix[1, 1] / np.sum(conf_matrix[:, 1])
recall = conf_matrix[1, 1] / np.sum(conf_matrix[1, :])
f1_score = 2 * precision * recall / (precision + recall)


import plotly.graph_objs as go
from plotly.subplots import make_subplots

fig = make_subplots(rows=1, cols=2, subplot_titles=("Confusion Matrix", "Precision/Recall/F1-Score"))

conf_matrix_fig = go.Figure(data=[go.Pie(labels=['Survived', 'Died'],
                                          values=[conf_matrix[0][0], conf_matrix[1][1]])])
conf_matrix_fig.update_layout(title='Confusion Matrix')
conf_matrix_fig.show()

conf_matrix = go.Heatmap(z=conf_matrix,
                         x=['Predicted Negative', 'Predicted Positive'],
                         y=['Actual Negative', 'Actual Positive'],
                         colorscale='Blues')
fig.add_trace(conf_matrix, row=1, col=1)

metrics = go.Bar(x=['Precision', 'Recall', 'F1-Score'],
                 y=[precision, recall, f1_score],
                 marker_color=['green', 'blue', 'orange'])
fig.add_trace(metrics, row=1, col=2)

fig.update_layout(height=500, width=1000, title_text="Adaboost Classifier Metrics on Titanic Dataset")
fig.show()


#  создаем subplots с заголовками
# для матрицы ошибок и метрик. Затем мы используем функцию go.Heatmap
# для создания heatmap с матрицей ошибок и функцию go.Bar для создания графиков
# для precision, recall и f1-score. Мы добавляем оба элемента в subplots и настраиваем
# layout и заголовок для всего графика. Используем функцию fig.show() для отображения графика.
#
# heatmap
# с матрицей ошибок и графики для precision, recall и f1-score.

