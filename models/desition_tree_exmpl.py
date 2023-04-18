import numpy as np
from typing import Optional


class Node:
    def __init__(self):
        self.right = None
        self.left = None
        self.split_ind = None
        self.split_val = None
        self.terminal_node = None

class DT:

    def __init__(self, max_depth, min_entropy:float=0, min_elem=0, max_nb_thresholds:int=10):
        self.max_depth = max_depth
        self.min_entropy = min_entropy
        self.min_elem = min_elem
        self.max_nb_thresholds = max_nb_thresholds
        self.root = Node()

    def train(self, inputs, targets):
        entropy_val = self.__shannon_entropy(targets,len(targets))
        self.__nb_dim = inputs.shape[1]
        self.__all_dim = np.arange(self.__nb_dim)

        self.__get_axis, self.__get_threshold = self.__get_all_axis, self.__generate_all_threshold
        self.__build_tree(inputs, targets, self.root, 1, entropy_val)

    def __get_random_axis(self):
        return np.random.randint(0, self.__nb_dim)

    def __get_all_axis(self):
        return self.__all_dim

    def __create_term_arr(self, target):
        return np.mean(target) if len(target) > 0 else 0

    def __generate_all_threshold(self, inputs):
        return np.linspace(np.min(inputs), np.max(inputs), num=self.max_nb_thresholds)

    def __generate_random_threshold(self, inputs):
        return np.random.uniform(np.min(inputs), np.max(inputs), size=self.max_nb_thresholds)

    @staticmethod
    def __disp(targets):
        return np.std(targets)

    @staticmethod
    def __shannon_entropy(targets, N):
        classes, counts = np.unique(targets, return_counts=True)
        probs = counts / N
        entropy = -np.sum(probs * np.log2(probs))
        return entropy

    def __inf_gain(self, targets_left, targets_right, node_disp, N):
        N_left = len(targets_left)
        N_right = len(targets_right)
        disp_left = self.__disp(targets_left)
        disp_right = self.__disp(targets_right)

        inf_gain = node_disp - (N_left / N) * disp_left - (N_right / N) * disp_right
        return inf_gain, disp_left, disp_right

    def __build_splitting_node(self, inputs, targets, entropy, N):
        max_inf_gain = -np.inf
        ax_max = None
        tay_max = None
        ind_left_max = None
        ind_right_max = None
        disp_left_max = None
        disp_right_max = None

        for ax in self.__get_axis():
            thresholds = self.__get_threshold(inputs[:, ax])
            for tay in thresholds:
                ind_left = np.where(inputs[:, ax] < tay)[0]
                ind_right = np.where(inputs[:, ax] >= tay)[0]
                if len(ind_left) == 0 or len(ind_right) == 0:
                    continue
                targets_left = targets[ind_left]
                targets_right = targets[ind_right]
                inf_gain, disp_left, disp_right = self.__inf_gain(targets_left, targets_right, self.__disp(targets), N)
                if inf_gain > max_inf_gain:
                    max_inf_gain = inf_gain
                    ax_max = ax
                    tay_max = tay
                    ind_left_max = ind_left
                    ind_right_max = ind_right
                    disp_left_max = disp_left
                    disp_right_max = disp_right

        return ax_max, tay_max, ind_left_max, ind_right_max, disp_left_max, disp_right_max

    def __build_tree(self, inputs, targets, node, depth, entropy):

        N = len(targets)
        if depth >= self.max_depth or entropy <= self.min_entropy or N <= self.min_elem:
            node.terminal_node = self.__create_term_arr(targets)
        else:

            ax_max, tay_max, ind_left_max, ind_right_max, disp_left_max, disp_right_max = self.__build_splitting_node(inputs, targets, entropy, N)
            node.split_ind = ax_max
            node.split_val = tay_max
            node.left = Node()
            node.right = Node()
            self.__build_tree(inputs[ind_left_max], targets[ind_left_max], node.left, depth + 1, disp_left_max)
            self.__build_tree(inputs[ind_right_max], targets[ind_right_max], node.right, depth + 1, disp_right_max)

    def get_predictions(self, inputs):
        predictions = []
        for input_vector in inputs:
            node = self.root
            while node.terminal_node is None:
                if input_vector[node.split_ind] < node.split_val:
                    node = node.left
                else:
                    node = node.right
            predictions.append(node.terminal_node)
        return np.array(predictions)




# Загрузим данные.
# Разобьем данные на обучающие, валидационные и тестовые наборы.
# Обучим дерево решений на обучающем наборе.
# Оценим качество модели на валидационном и тестовом наборах.

import pandas as pd
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, mean_squared_error

# Загрузим набор данных digits
digits = load_digits()
X_digits, y_digits = digits.data, digits.target

# Разделим датасет digits на обучающую, валидационную и тестовую выборки
X_train_digits, X_test_digits, y_train_digits, y_test_digits = train_test_split(X_digits, y_digits, test_size=0.2, random_state=42)
X_train_digits, X_val_digits, y_train_digits, y_val_digits = train_test_split(X_train_digits, y_train_digits, test_size=0.25, random_state=42)

# Создадим и обучим дерево решений для классификации
dt_classifier = DT(max_depth=10, min_entropy=0.1, min_elem=5)
dt_classifier.train(X_train_digits, y_train_digits)

# Оценим качество модели на валидационной и тестовой выборках для задачи классификации
y_val_pred_digits = np.round(dt_classifier.get_predictions(X_val_digits)).astype(int)
y_test_pred_digits = np.round(dt_classifier.get_predictions(X_test_digits)).astype(int)

accuracy_val_digits = accuracy_score(y_val_digits, y_val_pred_digits)
accuracy_test_digits = accuracy_score(y_test_digits, y_test_pred_digits)

conf_matrix_val_digits = confusion_matrix(y_val_digits, y_val_pred_digits)
conf_matrix_test_digits = confusion_matrix(y_test_digits, y_test_pred_digits)

print(f"Accuracy на валидационной выборке: {accuracy_val_digits:.4f}")
print(f"Accuracy на тестовой выборке: {accuracy_test_digits:.4f}")

print("\nConfusion Matrix на валидационной выборке:")
print(conf_matrix_val_digits)

print("\nConfusion Matrix на тестовой выборке:")
print(conf_matrix_test_digits)

# Загрузим набор данных wine quality
wine_quality_df = pd.read_csv("wine-quality-white-and-red.csv", delimiter=',')
wine_quality_df = wine_quality_df.drop(['type'], axis=1)
wine_quality_df['quality'] = wine_quality_df['quality'].astype(float)

X_wine_quality = wine_quality_df.drop(['quality'], axis=1).values
y_wine_quality = wine_quality_df["quality"].values


# Разделим датасет wine quality на обучающую, валидационную и тестовую выборки
X_train_wine, X_test_wine, y_train_wine, y_test_wine = train_test_split(X_wine_quality, y_wine_quality, test_size=0.2, random_state=42)
X_train_wine, X_val_wine, y_train_wine, y_val_wine = train_test_split(X_train_wine, y_train_wine, test_size=0.25, random_state=42)

# Создадим и обучим дерево решений для регрессии
dt_regressor = DT(max_depth=10, min_entropy=0.1, min_elem=5)
dt_regressor.train(X_train_wine, y_train_wine)

# Оценим качество модели на валидационной и тестовой выборках для задачи регрессии
y_val_pred_wine = dt_regressor.get_predictions(X_val_wine)
y_test_pred_wine = dt_regressor.get_predictions(X_test_wine)

mse_val_wine = mean_squared_error(y_val_wine, y_val_pred_wine)
mse_test_wine = mean_squared_error(y_test_wine, y_test_pred_wine)




print(f"\nMean Squared Error на валидационной выборке: {mse_val_wine:.4f}")
print(f"Mean Squared Error на тестовой выборке: {mse_test_wine:.4f}")




#  обучили дерево
# решений для задачи классификации
# (набор данных digits) и задачи регрессии
# (набор данных wine quality). Вывели на экран метрики
# качества модели: accuracy и confusion matrix для классификации,
# а также Mean Squared Error для регрессии.

# Теперь мы можем визуализировать результаты.
# Для этого используем heatmap для отображения
# confusion matrix и bar plot для отображения ошибок MSE.





import plotly.graph_objects as go
import plotly.subplots as sp
import plotly.graph_objects as go


# Confusion Matrix Heatmap для набора данных digits
hover_text = [[f"True: {i}<br>Predicted: {j}<br>Count: {conf_matrix_test_digits[i, j]}" for j in range(10)] for i in range(10)]

fig_digits_cm = go.Figure(go.Heatmap(z=conf_matrix_test_digits,
                                     x=list(range(10)),
                                     y=list(range(10)),
                                     text=hover_text,
                                     hoverinfo='text',
                                     colorscale='Viridis',
                                     showscale=True))

fig_digits_cm.update_layout(
    title="Confusion Matrix для тестовой выборки (digits)",
    xaxis_title="Predicted Label",
    yaxis_title="True Label",
    xaxis=dict(tickmode='array',
               tickvals=list(range(10)),
               ticktext=[str(i) for i in range(10)]),
    yaxis=dict(tickmode='array',
               tickvals=list(range(10)),
               ticktext=[str(i) for i in range(10)]),
    template="plotly_dark",
    autosize=False,
    width=600,
    height=600,
    margin=dict(l=50, r=50, b=100, t=100)
)

fig_digits_cm.show()








# Bar plot для ошибок MSE для набора данных wine quality
fig_mse = sp.make_subplots(rows=1, cols=2, specs=[[{'type': 'bar'}, {'type': 'bar'}]])

fig_mse.add_trace(go.Bar(x=["Validation", "Test"],
                         y=[mse_val_wine, mse_test_wine],
                         name="MSE (Wine Quality)"),
                 row=1, col=1)

fig_mse.update_layout(title="Mean Squared Error для Wine Quality",
                      xaxis_title="Выборки",
                      yaxis_title="MSE")

fig_mse.show()




# Теперь есть визуализация confusion matrix
# для задачи классификации (набор данных digits)
# и bar plot для ошибок MSE в задаче регрессии
# (набор данных wine quality).