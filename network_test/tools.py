import math
import torch
from torch.autograd import Variable


def quantize_data(output: torch.tensor):
    if False:
        return torch.quantize_per_tensor(output, 0.1, 0, torch.quint8).int_repr().tolist()
    return output.tolist()


def shorten_data(output: torch.tensor):
    return [[math.floor(x * 100) / 100.0 for x in row] for row in output.tolist()]


def r2_loss(prediction, target):
    target_mean = torch.mean(target)
    ss_tot = torch.sum((target - target_mean) ** 2)
    ss_res = torch.sum((target - prediction) ** 2)
    r2 = 1 - ss_res / ss_tot
    return r2


def binary_accuracy(predicted_early_exit_score, true_early_exit, score_limit=0.7):
    sigmoid_scores = torch.sigmoid(predicted_early_exit_score)
    early_exit_prediction = sigmoid_scores > score_limit

    correct_results_sum = (early_exit_prediction & (true_early_exit == 1.)).sum()
    total_prediction = early_exit_prediction.sum()
    print("Total EE: ", total_prediction)
    accuracy = correct_results_sum / total_prediction
    accuracy = torch.round(accuracy * 10000) / 100

    return accuracy, "{:.4f}".format(total_prediction)


def get_early_exit(y_pred, threshold=0.7):
    sigmoidScores = torch.sigmoid(y_pred)
    ee = sigmoidScores > threshold
    return ee


def early_exit(prediction: torch.Tensor, target: torch.Tensor, acceptable_range: float):
    difference = torch.sub(prediction, target)
    difference = torch.abs(difference)
    return torch.where(difference < acceptable_range, 1.0, 0.0)


def convert_tensor(tensor: torch.Tensor):
    if torch.cuda.is_available():
        tensor = tensor.cuda()
    return Variable(tensor)
