
from config.logistic_regression_config import cfg,choice_weights_init_type
from datasets.digits_dataset import Digits
from utils.visualisation import plot_accuracies_and_losses, plot_train_and_validation_accuracy, \
    plot_train_and_validation_loss
from models.logistic_regression_model import LogReg
from utils.enums import SetType
from utils.metrics import accuracy, confusion_matrix


train_accuracies = []
valid_accuracies = []
train_loss = []
valid_loss = []


def train_and_validate_logreg():
    choice_weights_init_type()
    dataset = Digits(cfg)
    logreg = LogReg(cfg, dataset.k, dataset.d)
    train_data = dataset(SetType.train)
    valid_data = dataset(SetType.valid)


    for epoch in range(cfg.nb_epoch):
        logreg.train(train_data['inputs'], train_data['onehotencoding'],
                     inputs_valid=valid_data['inputs'], targets_valid=valid_data['onehotencoding'])

        train_preds = logreg(train_data['inputs'])
        valid_preds = logreg(valid_data['inputs'])

        train_acc = accuracy(train_preds, train_data['targets'])
        valid_acc = accuracy(valid_preds, valid_data['targets'])

        train_accuracies.append(train_acc)
        valid_accuracies.append(valid_acc)

        train_conf_matrix = confusion_matrix(train_preds, train_data['targets'], dataset.k)
        valid_conf_matrix = confusion_matrix(valid_preds, valid_data['targets'], dataset.k)

        train_l_model = logreg._LogReg__target_function_value(train_data['inputs'], train_data['onehotencoding'])
        valid_l_model = logreg._LogReg__target_function_value(valid_data['inputs'], valid_data['onehotencoding'])




        train_loss.append(train_l_model)
        valid_loss.append(valid_l_model)

        print(f"Epoch {epoch + 1}/{cfg.nb_epoch}")
        print("Training Loss: {:.6f}, Validation Loss: {:.6f}".format(train_l_model, valid_l_model))
        print("Training Accuracy: {:.6f}, Validation Accuracy: {:.6f}".format(train_acc, valid_acc))
        print("Training Confusion Matrix:\n", train_conf_matrix)
        print("Validation Confusion Matrix:\n", valid_conf_matrix)
        print("")

    return train_accuracies, valid_accuracies, train_loss, valid_loss

if __name__ == "__main__":
    train_accuracies, valid_accuracies, train_loss, valid_loss = train_and_validate_logreg()
    plot_accuracies_and_losses( train_accuracies, valid_accuracies, train_loss, valid_loss)
    plot_train_and_validation_accuracy(train_accuracies, valid_accuracies)
    plot_train_and_validation_loss(train_loss, valid_loss)






































# import numpy as np
# from config.logistic_regression_config import cfg
# from datasets.digits_dataset import Digits
# from models.logistic_regression_model import LogisticRegression
# from utils.enums import SetType
# from utils.metrics import accuracy
# from utils.visualisation import plot_training_loss_and_accuracy,visualize_predictions
#
#
#
#
# if __name__ == '__main__':
#     # Load dataset
#     dataset = Digits(cfg)
#     train_data = dataset(SetType.train)
#     valid_data = dataset(SetType.valid)
#     test_data = dataset(SetType.test)
#
#     # Initialize logistic regression model
#     model = LogisticRegression(input_dim=train_data['inputs'].shape[1], num_classes=dataset.k,
#                                weights_init_type=cfg.weights_init_type, weights_init_kwargs=cfg.weights_init_kwargs)
#
#     # Train model
#     model.train(inputs=train_data['inputs'], targets=train_data['onehotencoding'], gamma=cfg.gamma,
#                 nb_epoch=cfg.nb_epoch, reg_lambda=0.001, valid_inputs=valid_data['inputs'],valid_targets=valid_data['onehotencoding'])
#
#     # Test model
#     test_predictions = model.predict(test_data['inputs'])
#     test_accuracy = accuracy(test_predictions, test_data['targets'])
#     print(f'Test Accuracy: {test_accuracy}')
#
#     # Plot training loss and accuracy
#     plot_training_loss_and_accuracy(model, train_data, valid_data)
#
#     # Save model
#     model.save('logistic_regression_model.pickle')
# s
#     # Load saved model
#     loaded_model = LogisticRegression.load('logistic_regression_model.pickle')
#
#     # Test loaded model
#     loaded_test_predictions = loaded_model.predict(test_data['inputs'])
#     loaded_test_accuracy = accuracy(loaded_test_predictions, test_data['targets'])
#     print(f'Loaded Test Accuracy: {loaded_test_accuracy}')
#
#     # Visualize predictions
#     visualize_predictions(loaded_model, test_data)
#
#
#
#


