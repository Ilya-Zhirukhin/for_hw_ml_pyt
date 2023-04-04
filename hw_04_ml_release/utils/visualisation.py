
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from config.logistic_regression_config import cfg


def plot_accuracies_and_losses(train_accuracies, valid_accuracies, train_loss, valid_loss):
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Accuracy", "Loss"))

    fig.add_trace(go.Scatter(x=list(range(1, cfg.nb_epoch + 1)), y=train_accuracies, mode='lines', name='Training Accuracy'), row=1, col=1)
    fig.add_trace(go.Scatter(x=list(range(1, cfg.nb_epoch + 1)), y=valid_accuracies, mode='lines', name='Validation Accuracy'), row=1, col=1)

    fig.add_trace(go.Scatter(x=list(range(1, cfg.nb_epoch + 1)), y=train_loss, mode='lines', name='Training Loss'), row=1, col=2)
    fig.add_trace(go.Scatter(x=list(range(1, cfg.nb_epoch + 1)), y=valid_loss, mode='lines', name='Validation Loss'), row=1, col=2)

    fig.update_layout(title="Training and Validation Accuracy and Loss", xaxis_title='Epochs', showlegend=True)

    fig.show()

def plot_train_and_validation_accuracy(train_accuracies,valid_accuracies):
    # Create a plot for the train and validation accuracy
    fig1 = go.Figure()

    fig1.add_trace(go.Scatter(x=list(range(len(train_accuracies))),
                              y=train_accuracies,
                              mode='lines',
                              name='Train Accuracy'))
    fig1.add_trace(go.Scatter(x=list(range(len(valid_accuracies))),
                              y=valid_accuracies,
                              mode='lines',
                              name='Validation Accuracy'))
    fig1.update_layout(title='Train and Validation Accuracy',
                       xaxis_title='Epoch',
                       yaxis_title='Accuracy')
    fig1.show()

def plot_train_and_validation_loss(train_loss,valid_loss):
    # Create a plot for the train and validation loss
    fig2 = go.Figure()

    fig2.add_trace(go.Scatter(x=list(range(len(train_loss))),
                              y=train_loss,
                              mode='lines',
                              name='Train Loss'))
    fig2.add_trace(go.Scatter(x=list(range(len(valid_loss))),
                              y=valid_loss,
                              mode='lines',
                              name='Validation Loss'))
    fig2.update_layout(title='Train and Validation Loss',
                       xaxis_title='Epoch',
                       yaxis_title='Loss')
    fig2.show()



