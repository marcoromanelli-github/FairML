import torch
from typing import Callable
import torch.nn.functional as F


def norml2(logits: torch.Tensor, temperature: float = 1.):
    """
    Compute the norml2 of logits
    :param logits: tensor of size (batch_size, num_classes)
    :param temperature: temperature for softmax scaling
    :return: norml2 of logits
    """
    return torch.sum(F.softmax(logits / temperature, dim=1) ** 2, dim=1)


def doctor(logits: torch.Tensor, temperature: float = 1.):
    """
    Compute the doctor score of logits
    :param logits: tensor of size (batch_size, num_classes)
    :param temperature: temperature for softmax scaling
    :return: doctor score of logits
    """
    g = norml2(logits, temperature)
    return (1 - g) / g


def acceptance_and_rejection_region(scores: torch.Tensor, threshold: float = 0.5,
                                    comparison_function: Callable = torch.gt):
    """
    Compute the acceptance and rejection region
    :param scores: tensor of size (batch_size, num_classes)
    :param threshold: threshold for doctor score
    :param comparison_function: callable function for comparison
    :return: acceptance and rejection region
    """

    # return a torch tensor of size (batch_size,) with 1 if the doctor score is greater than the threshold and 0
    # otherwise
    return comparison_function(scores, threshold).float()
