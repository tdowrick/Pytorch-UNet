import torch
from torch import Tensor
import numpy as np
import torch.nn.functional as F

def directed_distance_between_point_sets(A: np.ndarray, B: np.ndarray):
    """
    Computes the distance from point set A to B.

    :param A: an (n, d) NumPy array representing a set of points.
    :param B: an (m, d) NumPy array representing a set of points.
    :return: The distance from point set A to B.
    """
    num_A = A.shape[0]
    num_B = B.shape[0]
    stacked_B = np.vstack([B] * num_A)
    repeat_A = np.repeat(A, num_B, axis=0)

    diff = stacked_B - repeat_A
    distances = np.linalg.norm(diff, axis=1).reshape(-1, num_B)
    mins = np.min(distances, axis=1)
    sum_of_mins = np.sum(mins)

    return sum_of_mins / num_A


def compute_average_hausdorff_distance(A: np.ndarray, B: np.ndarray):
    """
    Computes the modified Hausdorff distance between point set A and B.

    :param A: an (n, d) NumPy array representing a set of points.
    :param B: an (m, d) NumPy array representing a set of points.
    :return: the modified Hausdorff distance between point set A and B.
    """
    # Compute the distances from A to B, and from B to A.
    distance_A = directed_distance_between_point_sets(A, B)
    distance_B = directed_distance_between_point_sets(A=B, B=A)

    return np.sum([distance_A, distance_B]) / 2

def directed_distance_between_tensors(A: Tensor, B: Tensor):
    num_A = A.shape[0]
    num_B = B.shape[0]

    stacked_B = torch.vstack([B] * num_A)
    repeat_A = torch.repeat_interleave(A, num_B, axis=0)

    diff = stacked_B - repeat_A
    distances = torch.linalg.norm(diff.double(), dim=1).reshape(-1, num_B)
    mins = torch.min(distances, 1)
    sum_of_mins = torch.sum(mins[0])
    
    return sum_of_mins / num_A

def average_hausdorff_distance(input: Tensor, target: Tensor):
    x = directed_distance_between_tensors(input, target)
    y = directed_distance_between_tensors(target, input)
    return (x + y)/2

def get_coordinates_of_index(input: Tensor, index: int):
    """ return x, y coordinates of all points that are equal to index"""
    x, y = torch.where(input==index)

    return torch.stack((x, y), dim=1)

def hausdorff_loss(input: Tensor, target: Tensor):

    hausdorff = 0

    num_in_batch = input.shape[0]
    for j in range(num_in_batch):

        probs = torch.sigmoid(input)[j]
        one_hot = F.one_hot(probs.argmax(dim=0), 3).permute(2, 0, 1) #TODO variable class number
        pred_mask = torch.argmax(one_hot, dim=0)

        # TODO: Should the hausdorff distance be averaged across all classes?
        # Bongjin took the sum of all classes, as done here

        # TODO: How to handle the case where no pixels are detectd in the input?
        # Just continuing at the moment. Should there be a fixed penalty for this?
        for i in range(1, torch.max(target)):

            if i not in target[j,:,:]:
                continue

            if i not in pred_mask:
                continue

            input_pixels = get_coordinates_of_index(pred_mask, i)
            target_pixels = get_coordinates_of_index(target[j,:,:], i)

            hausdorff += average_hausdorff_distance(input_pixels, target_pixels)

    return hausdorff

def test_compute_hausdorff_distance_valid():
    """
    Test for computing the modified Hausdorff distance.it()

    """
    A = np.array([[0, 1], [2, 3]])
    B = np.array([[4, 5], [6, 7], [8, 9]])

    A = torch.from_numpy(A)
    B = torch.from_numpy(B)

    distance = average_hausdorff_distance(A, B)

test_compute_hausdorff_distance_valid()