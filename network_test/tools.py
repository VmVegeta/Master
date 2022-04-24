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


def binary_acc(y_pred, y_test):
    sigmoidScores = torch.sigmoid(y_pred)
    y_pred_tag = sigmoidScores > 0.7

    correct_results_sum = ((y_pred_tag == True) & (y_test == 1.)).sum()
    total_prediction = (y_pred_tag == True).sum()
    print("Total EE: ", total_prediction)
    acc = correct_results_sum / total_prediction
    acc = torch.round(acc * 10000) / 100

    return acc, "{:.4f}".format(total_prediction)


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
