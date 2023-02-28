import plotly.graph_objects as go
import numpy as np
import plotly.graph_objects as go
import numpy as np
from models import linear_regression_model

class Visualisation():
    @staticmethod
    def visualise_predicted_trace(prediction: np.ndarray, inputs: np.ndarray, targets: np.ndarray, plot_title=''):
        # fig = go.Figure()
        # fig.add_trace(go.Scatter(x=inputs.flatten(), y=targets.flatten(), mode='markers', name='Target'))
        # fig.add_trace(go.Scatter(x=inputs.flatten(), y=prediction.flatten(), mode='lines', name='Model Prediction'))
        # fig.update_layout(title=plot_title, xaxis_title='Inputs', yaxis_title='Outputs',
        #                   legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01))
        # fig.show()
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=inputs.flatten(), y=targets.flatten(), mode='markers', name='Target'))
        fig.add_trace(go.Scatter(x=inputs.flatten(), y=prediction.flatten(), mode='lines', name='Model Prediction'))
        fig.update_layout(title=plot_title, xaxis_title='Inputs', yaxis_title='Outputs',
                          legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01))

        # adjust the range of the y-axis to make sure the model prediction line passes through the target points
        y_range = [min(targets.min(), prediction.min()), max(targets.max(), prediction.max())]
        fig.update_layout(yaxis=dict(range=y_range))

        fig.show()
    # @staticmethod
    # def visualise_predicted_trace(prediction: np.ndarray, inputs: np.ndarray, targets: np.ndarray, plot_title=''):
    #     # plot the predicted and target traces
    #     trace_pred = go.Scatter(x=inputs, y=prediction, mode='lines', name='Prediction')
    #     trace_target = go.Scatter(x=inputs, y=targets, mode='markers', name='Target')
    #     fig = go.Figure([trace_pred, trace_target])
    #
    #     # update the plot layout
    #     fig.update_layout(title=plot_title, xaxis_title='Input', yaxis_title='Output',
    #                       legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01))
    #     fig.show()
    # #
    # @staticmethod
    # def create_plots(models):
    #     for model in models:
    #         # Generate inputs for prediction
    #         inputs = np.linspace(-1, 1, 100)
    #
    #         # Generate targets for inputs
    #         targets = model(inputs)
    #
    #         # Calculate model prediction
    #         prediction = model.predict(inputs)
    #
    #         # Calculate model error
    #         mse = model.calculate_error(inputs, targets)
    #
    #         # Create plot title
    #         plot_title = f"Max Degree: {model.max_degree}, MSE: {mse:.2f}"
    #
    #         # Visualize predicted trace and targets
    #         Visualisation.visualise_predicted_trace(prediction, inputs, targets, plot_title)
    @staticmethod
    def visualise_error():
        pass






# model1 = linear_regression_model.LinearRegression(base_functions=[lambda x: x])
# model2 = linear_regression_model.LinearRegression(base_functions=[lambda x: x ** i for i in range(9)])
# model3 = linear_regression_model.LinearRegression(base_functions=[lambda x: x ** i for i in range(101)])
#
# Visualisation.create_plots([model1, model2, model3])

