from models.linear_regression_model import LinearRegression
from datasets.linear_regression_dataset import LinRegDataset
from utils.metrics import MSE
from utils.visualisation import Visualisation





def experiment(lin_reg_cfg, visualise_prediction=True):
    # lin_reg_model = LinearRegression(lin_reg_cfg.base_functions)
    # linreg_dataset = LinRegDataset()(lin_reg_cfg.dataframe_path)
    #
    # predictions = lin_reg_model(linreg_dataset['inputs'])
    # error = MSE(predictions, linreg_dataset['targets'])
    #
    # if visualise_prediction:
    #     Visualisation.visualise_predicted_trace(predictions,
    #                                             linreg_dataset['inputs'],
    #                                             linreg_dataset['targets'],
    #                                             plot_title=f'Полином степени {len(lin_reg_cfg.base_functions)}; MSE = {error}')

    # for degree in [1, 8, 100]:
    #     lin_reg_model = LinearRegression(lin_reg_cfg.base_functions[:degree])
    #     linreg_dataset = LinRegDataset()(lin_reg_cfg.dataframe_path)
    #
    #     predictions = lin_reg_model(linreg_dataset['inputs'])
    #     error = MSE(predictions, linreg_dataset['targets'])
    #
    #     if visualise_prediction:
    #         Visualisation.visualise_predicted_trace(predictions,
    #                                                 linreg_dataset['inputs'],
    #                                                 linreg_dataset['targets'],
    #                                                 plot_title=f'Polynomial Degree: {degree}, Error: {round(error, 2)}')
    for degree in [1, 8, 100]:
        lin_reg_cfg.base_functions = [lambda x, i=i: x ** i for i in range(degree + 1)]
        lin_reg_model = LinearRegression(lin_reg_cfg.base_functions)
        linreg_dataset = LinRegDataset()(lin_reg_cfg.dataframe_path)

        predictions = lin_reg_model(linreg_dataset['inputs'])
        error = round(MSE(predictions, linreg_dataset['targets']), 2)

        title = f'Polynomial degree {degree}; MSE = {error}'
        Visualisation.visualise_predicted_trace(predictions, linreg_dataset['inputs'], linreg_dataset['targets'], title)
if __name__ == '__main__':
    from configs.linear_regression_cfg import cfg as lin_reg_cfg
    experiment(lin_reg_cfg,visualise_prediction=True)

