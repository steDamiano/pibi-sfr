# This script includes portions of code from the repository "PIDL-sound-field-reconstruction"
# (https://github.com/steDamiano/PIDL-sound-field-reconstruction/), which is licensed under the MIT License.

# The original license text is included below.

# Copyright (c) 2025 Stefano Damiano

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# This script includes portions of code from the repository "local_soundfield_reconstruction"
# (https://github.com/manvhah/local_soundfield_reconstruction), which is licensed under the MIT License.
# The original license text is included below.

# Copyright (c) 2021 manvhah

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import numpy as np
import torch
from scipy.spatial import distance
from scipy.special import spherical_jn
import matplotlib.pyplot as plt
from pathlib import Path

def NMSE(u_estimated: torch.Tensor, u_ground_truth: torch.Tensor) -> torch.Tensor:
    '''
    Normalized Mean Squared Error (NMSE) between estimated and ground truth sound fields.
    
    Parameters
    ----------
    u_estimated: torch.Tensor
        Estimated sound field.
    u_ground_truth: torch.Tensor
        Ground truth sound field.
    
    Returns
    -------
    NMSE: torch.Tensor
        NMSE value.
    '''
    return (torch.abs(u_estimated - u_ground_truth) ** 2) / (torch.norm(u_ground_truth, p=2) ** 2)

def NCC(u_estimated: torch.Tensor, u_ground_truth: torch.Tensor) -> torch.Tensor:
    '''
    Normalized Cross-Correlation (NCC) between estimated and ground truth sound fields.
    
    Parameters
    ----------
    u_estimated: torch.Tensor
        Estimated sound field.
    u_ground_truth: torch.Tensor
        Ground truth sound field.
    
    Returns
    -------
    NCC: torch.Tensor
        NCC value.
    '''
    u_estimated = u_estimated.ravel()
    u_ground_truth = u_ground_truth.ravel()
    return torch.abs(u_estimated @ torch.conj(u_ground_truth)) / (torch.norm(u_estimated, p=2) * torch.norm(u_ground_truth, p=2))

def sinc_kernel_multifreq(psize: tuple, K: int, frequencies: list | np.ndarray, meas_dist: float) -> np.ndarray:
    '''
    Generates a dictionary of sinc functions for multiple frequencies.
    
    Parameters
    ----------
    psize: tuple
        Size of the grid (width, height).
    K: int
        Number of atoms.
    frequencies: list or np.ndarray
        Frequencies for which the sinc functions are generated.
    meas_dist: float
        Space between adjacent grid points.
    
    Returns
    -------
    H: np.ndarray
        Dictionary of sinc functions, shape (N, K), where N is the number of grid points and K the number of atoms.
    '''
    x = np.arange(psize[0])
    y = np.arange(psize[1])
    N = psize[0] * psize[1]
    mu = np.zeros(N)
    xx, yy = np.meshgrid(x,y)
    x_grid = np.array([z for z in zip(xx.flatten(),yy.flatten())])
    H = np.zeros((N,K))
    for k in range(K):
        print(f'Generating sinc dictionary, atom: {k}/{K}')
        phasedist = 2*np.pi*frequencies[k]/343 * meas_dist * distance.cdist(x_grid,x_grid)
        Sigma  = spherical_jn(0, phasedist)

        H[:,k] = np.random.multivariate_normal(mu,Sigma)
        H[:, k] *= 1 / np.linalg.norm(H[:, k])
    np.save(f'sinc_dictionary_{frequencies[0]}_{frequencies[-1]}.npy', H)
    return H

def plot_sound_field(n_grid_points: int, estimated: np.ndarray, ground_truth: np.ndarray, nmse: float, path: Path, frequency: float = None) :
    '''
    Plot the estimated and ground truth sound fields side by side.
    Parameters
    ----------
    n_grid_points: int
        Number of grid points along each dimension.
    estimated: np.ndarray
        Estimated sound field.
    ground_truth: np.ndarray
        Ground truth sound field.
    nmse: float
        Normalized Mean Squared Error (NMSE) value.
    path: Path
        Path to save the plot.
    frequency: float, optional
        Frequency of the sound field, used in the plot file name if present (default: None).
    '''
    _, ax = plt.subplots(1,2)
    ax[0].imshow(np.reshape(estimated, (n_grid_points, n_grid_points)), origin='lower')
    ax[0].set_title(f'Predicted: NMSE={nmse:.2f}')
    
    ax[1].imshow(np.reshape(ground_truth, (n_grid_points, n_grid_points)), origin='lower')
    ax[1].set_title('Ground Truth')
    
    if frequency is not None:
        plt.savefig(path.joinpath(f'sf_plot_{frequency}.png'))
    else:
        plt.savefig(path.joinpath('sf_plot.png'))
    
    plt.close()