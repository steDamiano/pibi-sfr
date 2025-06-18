# This script includes portions of code from the repository "PIBI-Net"
# (https://github.com/MonikaNagy-Huber/PIBI-Net), which is licensed under the MIT License.

# Copyright (c) 2023 nagymo

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

from torch import nn
import torch
import numpy as np
from scipy.special import hankel1
from src.utils import NMSE, NCC

class PIBIModel(nn.Module):
    '''
    PIBI-Net method for sound field reconstruction, based on a boundary integral neural network.
    This method estimates the sound field at given coordinates within a closed region,by solving the Kirchhoff-Helmholtz boundary integral equation.
    A neural network is used to approximate the boundary pressure density function, which is then used to compute the pressure within the region.
    
    Attributes
    ----------
    n_hidden_neurons: int
        Number of neurons in each hidden layer of the neural network.
    n_hidden_layers: int
        Number of hidden layers in the neural network.
    n_integration_points: int
        Number of integration points used for the computation of the boundary pressure density.
    temporal_frequency: float
        Temporal frequency at which sound field is estimated.
    region_size: np.ndarray
        Size of the 2D reconstruction region, given as a 2D array [width, height].
    lr: float
        Learning rate for the optimizer.
    iterations: int
        Number of training iterations.
    epsilon: float
        Small value to avoid division by zero and to ensure numerical stability.
    component: str
        Specifies whether to estimate the 'real' or 'imaginary' component of the sound field.

    Methods
    -------
    predict(x: torch.Tensor) -> torch.Tensor:
        Predicts the sound field at given set of coordinates using the trained model.
    train_model(coords_microphones: torch.Tensor, train_data: torch.Tensor) -> list:
        Train the model using the provided measurement microphone coordinates and corresponding (ground truth) pressure.
    evaluate(u_estimated: torch.Tensor, u_ground_truth: torch.Tensor) -> tuple:
        Evaluates performance using Normalized Mean Squared Error (NMSE) and Normalized Cross-Correlation (NCC) metrics.
    '''
    def __init__(
            self,
            n_hidden_neurons: int,
            n_hidden_layers: int,
            n_integration_points: int,
            temporal_frequency: float,
            region_size: np.ndarray,
            lr: float = 0.001,
            iterations: int = 5_001,
            epsilon: float = 0.1,
            component: str = 'real'
        ):
        
        super().__init__()
        activation = nn.Tanh
        
        self.n_int_points = n_integration_points
        self.temporal_frequency = temporal_frequency
        self.region_size = torch.tensor(region_size, dtype=torch.float32)
        self.lr = lr
        self.iterations = iterations
        self.component = component
        
        self.epsilon = epsilon
        self.x_int = self._integration_points()

        self.input_layer = nn.Sequential(*[
                nn.Linear(2, n_hidden_neurons),
                activation()
                ])
        
        self.hidden_layers = nn.Sequential(*[
                nn.Sequential(*[
                    nn.Linear(n_hidden_neurons, n_hidden_neurons),
                    activation()
                    ]) for _ in range(n_hidden_layers)
                ])
        
        self.output_layer = nn.Linear(n_hidden_neurons, 1)

        self.loss_function = torch.nn.MSELoss()
        self.optimiser = torch.optim.Adam(self.parameters(), lr=self.lr)

    def forward(self, input: torch.Tensor):
        # Map the input to [-1,1] --> The reconstruction region is a square between -1 and 1
        input = 2 * (input - torch.tensor([-1 - self.epsilon, -1 - self.epsilon])) / (torch.tensor([1 + self.epsilon, 1 + self.epsilon]) - torch.tensor([-1 - self.epsilon, -1 - self.epsilon])) - 1
        
        x = self.input_layer(input) 
        x = self.hidden_layers(x)
        x = self.output_layer(x)
        
        return x

    def _calc_potentials_inside(self, x: torch.Tensor, temporal_frequency: float):
        '''
        Calculates the single and double layer potentials inside the region, at positions x, given the boundary integration points. The region is assumed to be a square with side length 2m.

        Parameters
        ----------
        x: torch.Tensor
            Coordinates where the sound field is estimated.
        temporal_frequency: float
            Temporal frequency at which the sound field is estimated.
        
        Returns
        -------
        single_layer: torch.Tensor
            Single layer potential at the coordinates x.
        double_layer: torch.Tensor
            Double layer potential at the coordinates x.
        '''
        # The considered region size is 2m --> this will be changed in the future
        side = 1 + 1 * self.epsilon

        # Retrieve boundary points and inward normal
        y = self.x_int
        normal_y = self._inward_normal(y)

        # Compute the boundary density at integration points y using the neural network
        h_y = self(y) 

        # Compute the single layer potential using the fundamental solution
        G = self._fundamental_solution(x, y, 2 * np.pi * temporal_frequency /343)
        
        grad_ones = torch.ones_like(h_y)
        dh_dy = torch.autograd.grad(h_y, y, grad_outputs=grad_ones, create_graph=True)[0]
        dh_dn = torch.sum(dh_dy * normal_y, dim=1).view(-1, 1)
        
        single_layer = 4 * side * torch.mean(dh_dn * G, dim=0)

        # Compute the double layer potential using the gradient of the fundamental solution
        dG_dy = self._gradient_fundamental(x, y, 2 * np.pi * temporal_frequency / 343)

        dG_dn = torch.sum(dG_dy * normal_y[:, None, :], dim=2)
        double_layer = 4 * side * torch.mean(dG_dn * h_y, dim=0)

        return single_layer.squeeze(), double_layer.squeeze()

    def predict(self, x: torch.Tensor):
        '''
        Predicts the sound field at given coordinates x using the trained model.
        
        Parameters
        ----------
        x: torch.Tensor
            2D Cartesian coordinates where the sound field is estimated, shape (N, 2), where N is the number of points.
        Returns
        -------
        u_int_data: torch.Tensor
            Estimated sound field at the coordinates x computed using the Kirchhoff-Helmholtz BIE, shape (N, 1).
        '''
        single_layer, double_layer = self._calc_potentials_inside(x, self.temporal_frequency)
        u_int_data = (single_layer.squeeze() - double_layer.squeeze()).view(-1, 1)
        
        return u_int_data
    
    def train_model(
            self,
            coords_microphones: torch.Tensor,
            train_data: torch.Tensor,
        ):
        '''
        Trains PIBI-Net using the provided microphone coordinates and corresponding (ground truth) pressure data.
        
        Parameters
        ----------
        coords_microphones: torch.Tensor
            Coordinates of the microphones where the sound field is measured, shape (M, 2), where M is the number of microphones.
        train_data: torch.Tensor
            Ground truth pressure at the M available measurement microphones, shape (M, 1).
        
        Returns
        -------
        loss_values: list
            List of loss values recorded during training.
        '''

        loss_values = []
        
        for iteration in range(self.iterations):
            self.optimiser.zero_grad()

            predicted_sound_field = self.predict(coords_microphones)
            
            loss = self.loss_function(predicted_sound_field, train_data.view(-1,1))
            loss_values.append(loss.item())

            loss.backward()
            self.optimiser.step()

        return loss_values

    def evaluate(
            self,
            u_estimated: torch.Tensor,
            u_ground_truth: torch.Tensor,
        ):
        '''
        Evaluates the performance of the model using Normalized Mean Squared Error (NMSE) and Normalized Cross-Correlation (NCC) metrics.
        
        Parameters
        ----------
        u_estimated: torch.Tensor
            Estimated sound field at the grid coordinates, shape (N, 1), where N is the number of points.
        u_ground_truth: torch.Tensor
            Ground truth sound field at the grid coordinates, shape (N, 1), where N is the number of points.
        
        Returns
        -------
        nmse: torch.Tensor
            Normalized Mean Squared Error (NMSE) between the estimated and ground truth pressure.
        ncc: torch.Tensor
            Normalized Cross-Correlation (NCC) between the estimated and ground truth pressure.
        '''
        nmse = NMSE(u_estimated.squeeze(), u_ground_truth)
        ncc = NCC(u_estimated.squeeze(), u_ground_truth)

        return nmse, ncc
    
    def _integration_points(self):
        '''
        Generates integration points to solve the boundary integral equation.
        The integration points are uniformly distributed along the boundary of a square region, with a small random perturbation.

        Returns
        -------
        x_int: torch.Tensor
            Integration points for the boundary integral equation, shape (n_integration_points, 2).
        '''
        points = torch.linspace(0, 4, self.n_int_points + 1)[:-1]
        points += torch.rand(points.shape) * (1 / self.n_int_points) + np.random.rand() * 0.05

        side_x = self.region_size[0] + self.region_size[0] * self.epsilon
        side_y = self.region_size[1] + self.region_size[1] * self.epsilon
        middle = torch.tensor((self.region_size[0] / 2, self.region_size[1] / 2)).view(1,2)
        points = torch.remainder(points, 4)
        v1 = torch.tensor((0, side_y))[None,]
        v2 = torch.tensor((side_x, 0))[None,]
        v3 = torch.tensor((0, -side_y))[None,]
        v4 = torch.tensor((-side_x, 0))[None,]
        
        points = points.view(-1,1)
        
        x_int = (middle - 0.5 * (v1 + v2) + torch.clamp(points,0,1) * v1 + torch.clamp(points-1,0,1) * v2 + torch.clamp(points-2,0,1) * v3 + torch.clamp(points-3,0,1) * v4).requires_grad_(True)

        x_int = x_int - torch.Tensor(self.region_size / 2).repeat((len(x_int), 1))
        
        return x_int

    def _inward_normal(self, x: torch.Tensor):
        '''
        Computes the inward normal vector at the boundary points x.
        
        Parameters
        ----------
        x: torch.Tensor
            Boundary points where the inward normal need to be computed, shape (N, 2), where N is the number of points.
        Returns
        -------
        normals: torch.Tensor
            Inward normal vectors at the boundary points x, shape (N, 2).
        '''

        normalized_x = x / self.region_size
        middle = torch.tensor((0,0)).view(1,2)
        max_indices = torch.argmax(torch.abs(normalized_x - middle), dim=-1, keepdim=False)
        normals = torch.zeros_like(x)
        sign = torch.sign(x)
        temp_range = torch.arange(x.size(0))
        normals[temp_range, max_indices] =  - sign[temp_range, max_indices]

        return normals

    def _fundamental_solution(
            self, 
            x: torch.Tensor, 
            y: torch.Tensor, 
            k: torch.Tensor
        ):
        '''
        Computes the fundamental solution of the Helmholtz equation, given the wave number k.
        The fundamental solution is given by the Hankel function of the first kind computed between source points y and receiver points x.

        Parameters
        ----------
        x: torch.Tensor
            Receiver coordinates, shape (N, 2), where N is the number of receiver points.
        y: torch.Tensor
            Source coordinates (i.e., boundary points), shape (M, 2), where M is the number of boundary points.
        k: torch.Tensor
            Wave number, shape (1,).
        
        Returns
        -------
        green_function: torch.Tensor
            Fundamental solution of the Helmholtz equation at the N points, 2D case, shape (N, 1).
        '''

        if x.shape == y.shape:
            distance =  torch.norm(y - x, p=2, dim=1).view(-1,1)
        else:
            distance = torch.cdist(y, x, p=2)

        if self.component == 'real':
            green_function = np.real(-1j / 4 *  hankel1(0, (k * distance).detach().numpy()))
        else:
            green_function = np.imag(-1j / 4 *  hankel1(0, (k * distance).detach().numpy()))

        return torch.tensor(green_function, dtype=torch.float32, requires_grad=True)

    def _gradient_fundamental(
            self, 
            x: torch.Tensor, 
            y: torch.Tensor, 
            k: torch.Tensor
        ):
        '''
        Computes the gradient of the fundamental solution of the Helmholtz equation, given the wave number k.
        The gradient is computed as the derivative of the Hankel function of the first kind with respect to the source points y.
        
        Parameters
        ----------
        x: torch.Tensor
            Receiver coordinates, shape (N, 2), where N is the number of receiver points.
        y: torch.Tensor
            Source coordinates (i.e., boundary points), shape (M, 2), where M is the number of boundary points.
        k: torch.Tensor
            Wave number, shape (1,).
        Returns
        -------
        grad_green: torch.Tensor
            Gradient of the fundamental solution of the Helmholtz equation at the N points, 2D case, shape (N, 2).
        '''

        if x.shape == y.shape:
            pairwise_diff = x - y
            distance = torch.norm(y-x, p=2, dim=1).view(-1,1)
        else:
            pairwise_diff = x[None,:,:] - y[:,None,:]
            distance = torch.cdist(y,x, p=2)
        if self.component == 'real':
            h1 = torch.tensor(hankel1(1, (k * distance).detach().numpy()))
            grad_green = torch.real((1j * k / 4) * h1[..., None] * (pairwise_diff / distance[...,None]))
        else:
            h1 = torch.tensor(hankel1(1, (k * distance).detach().numpy()))
            grad_green = torch.imag((1j * k / 4) * h1[..., None] * (pairwise_diff / distance[...,None]))
        
        return grad_green