"""

Author: Andrew Quintana
Email: aquintana7@gatech.edu
Version: 0.1.0
License: [License Name]

Usage:
Utility functions used by most components. 

Classes:
Component: super class for component classes

Functions:
approximately_equal

Attributes:
INFO: enum for communicating with fsm

Dependencies:
time, enum, typing, numpy

License:
[Include the full text of the license you have chosen for your code]

Examples:
[Provide some example code snippets demonstrating how to use the module/package]

"""

import time
import os
import cv2
from enum import Enum
import numpy as np
from numpy.typing import NDArray, ArrayLike
from numpy.linalg import *

# ------------------------------------- SYSTEM -------------------------------------

obstacle_ids = ["8"]

# common filepaths
dir = os.path.abspath(__file__)
proj_dir = os.path.abspath(os.path.join((dir), "../../../"))
docs_dir = os.path.join(proj_dir, "docs")
image_dir = os.path.join(docs_dir, "images")
cal_dir = os.path.join(image_dir, "calibration")

# debugging/logging
logging = True
debug = True
verbose = False

# ------------------------------------- CLASSES -------------------------------------

class Component:
    """
    Base component superclass.
    """

    def __init__( self, logging, debug, verbose ):
        """
        Constructor of superclass.

        Args:
            logging (bool): controls logging output
            debug (bool): controls debug output
            verbose (bool): controls verbose output
        """
        self.logging = logging
        self.debug = debug
        self.verbose = verbose

# ----------------------------------- ENUMS/TYPES -----------------------------------

# info enum for passing information to fsm updater
class INFO( Enum ):
    NA = 0
    GOAL_NOT_FOUND = 1
    GOAL_FOUND = 2
    AT_GOAL = 3
    NOT_AT_GOAL = 4 
    ERROR = -1
    UNKNOWN = -2
    FAILED = -3

# ------------------------------------- FUNCTIONS ------------------------------------

def approximately_equal( a, b, tol ):
    """
    Determine if the vectors are approximately similar, relative to the tolerance provided
    
    Args:
        a (state): comparison vector
        b (state): target vector
        tol (float): tolerance
    
    Returns:
        equal (bool): returns whether or not vectos are within tolerance of one another
    """

    # calculate the distance between vectors
    delta = norm(a[:2] - b[:2])

    # compare to tolerance
    equal = delta <= tol

    return equal

def deg_rad( deg ):
    """
    Convert degrees into radians
    
    Args:
        deg (float): degree to be converted to radians
    
    Returns:
        rad (float): representative radian value
    """
    return deg * (np.pi / 180)

def insert_rows_cols(matrix, row_index, col_index, num_rows, num_cols):
    """
    Inserts rows and columns into a NumPy matrix at the specified indices and replaces them with all 0s.
    
    Args:
        matrix (np.ndarray): The input matrix.
        row_index (int): The index at which to insert the new rows.
        col_index (int): The index at which to insert the new columns.
        num_rows (int): The number of rows to be inserted.
        num_cols (int): The number of columns to be inserted.
    
    Returns:
        np.ndarray: The modified matrix with the new rows and columns inserted and replaced with 0s.
    """
    if len(matrix.shape) == 1: matrix = np.expand_dims(matrix, axis=1)
    rows, cols = matrix.shape
    
    # Create new rows and columns filled with 0s
    new_rows = np.zeros((num_rows, cols))
    new_cols = np.zeros((rows + num_rows, num_cols))
    
    # Insert new rows and columns
    matrix = np.concatenate([matrix[:row_index], new_rows, matrix[row_index:]], axis=0)
    matrix = np.concatenate([matrix[:, :col_index], new_cols, matrix[:, col_index:]], axis=1)
    
    return matrix

def print_matrix( name, matrix ):
    """
    Credit: https://www.perplexity.ai/search/Im-trying-to-9IciM4_xQA.S5fKxmz9MPw#17

    Prints a 2D NumPy array in a clean, formatted way, including the matrix name.
    
    Args:
        name (str): The name of the matrix.
        matrix (np.ndarray): The 2D NumPy array to be printed.
    """
    rows, cols = matrix.shape
    
    # Determine the maximum width of each column
    col_widths = [max(len(str(matrix[i, j])) for i in range(rows)) for j in range(cols)]
    
    # Print the matrix name
    print(f"{name}:")
    
    # Print the matrix row by row
    for i in range(rows):
        row = [f"{matrix[i, j]:>{col_widths[j]}}" for j in range(cols)]
        print(" | ".join(row))
        
        if i == 0:
            print("-+-".join("-" * width for width in col_widths))

def print_state( env ):
    """
    Prints all environments at the current step
    
    Args:
        env (Dict[str,state]): List of states with keys for each string ID.
    """

    for key in env:
        print(f"ID: {key}\n\tX: {env[key][0]}\tY: {env[key][1]}\tPSI: {env[key][2]}")

def print_status( hyphen_sets, status ):
    """
    Prints current status with provided hyphen structure
    
    Args:
        hyphen_sets (int): number of hyphen blocks
        status (str): status to print
    """

    # provide structure
    hyphens = "- - - - - "
    full_hyphens = hyphens * hyphen_sets

    # print status with hyphen set and timestamp
    now = time.strftime("%H:%M:%S", time.localtime(time.time()))
    print(f"{now} - {full_hyphens}{status} - - - - - - - - - - - - - - -\n")

def save(matrix, dir=None, subdir_0=None, subdir_1=None, filename=None):
    """
    Simplifying the saving process to add folders for saving intermidate steps.

    Args:
        matrix (ndarray): matrix to be saved
        category (str): name of category of test
        step (str): name of step being tested
        subj (str): name of subject being processed
        filename (str): name of final file to be saved

    Returns:
        path (str): final pathname returned for live deletions

    NOTE: all strings above may be used to just generage directory layers
    """

    img = matrix.astype(np.uint8)

    path = os.path.join(docs_dir,"images")
    
    if dir != None: path = os.path.join(path,dir)

    if not os.path.exists(path):
        os.makedirs(path)

    if subdir_0 != None: path = os.path.join(path,subdir_0)

    if not os.path.exists(path):
            os.makedirs(path)

    if subdir_1 != None: path = os.path.join(path,subdir_1)

    if not os.path.exists(path):
        os.makedirs(path)

    path = os.path.join(path,filename)
    if verbose: print(f"\tImage {path} saved.")

    cv2.imwrite(path, img)

    return path

def get_psi_symbol():
    return 'ψ'

def compare_arrays(arr1, arr2):
    """
    Compares two NumPy arrays and prints the row, column, and value of each inconsistency.

    Args:
        arr1 (np.ndarray): The first array to compare.
        arr2 (np.ndarray): The second array to compare.

    Raises:
        ValueError: If the input arrays have different shapes.
    """
    if arr1.shape != arr2.shape:
        raise ValueError("Input arrays must have the same shape.")

    for i in range(arr1.shape[0]):
        for j in range(arr1.shape[1]):
            if arr1[i, j] != arr2[i, j]:
                print(f"Inconsistency at row {i}, column {j}: {arr1[i, j]} != {arr2[i, j]}")
