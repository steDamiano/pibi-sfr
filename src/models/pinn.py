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

import torch
from torch import nn
from src.utils import NMSE, NCC
import numpy as np

class PINNModel(nn.Module):
    '''
    Physics-Informed Neural Network (PINN) for sound field reconstruction.
    This model estimates the sound field at given coordinates within a closed region, by using a feedforward neural network that incorporates the Helmholtz equation in the training loss.
    
    Attributes
    ----------
    n_hidden_neurons: int
        Number of neurons in each hidden layer of the neural network.
    n_hidden_layers: int
        Number of hidden layers in the neural network.
    n_collocation_points: int
        Number of collocation points used for computing the Helmholtz equation during training.
    temporal_frequency: float
        Temporal frequency at which sound field is estimated.
    region_size: np.ndarray
        Size of the 2D reconstruction region, given as a 2D array [width, height].
    lambda_physics: float
        Weighting factor for the physics loss term.
    lr: float
        Learning rate for the optimizer.
    iterations: int
        Number of training iterations.
    epsilon: float
        Small value to avoid division by zero and to ensure numerical stability.
    
    Methods
    -------
    predict(x: torch.Tensor) -> torch.Tensor
        Predict the sound field at a given set of coordinates using the trained model.
    train_model(coords_microphones: torch.Tensor, train_data: torch.Tensor) -> list
        Train the model using the provided measurement microphone coordinates and corresponding (ground truth) pressure.
    evaluate(u_estimated: torch.Tensor, u_ground_truth: torch.Tensor) -> tuple
        Evaluates performance using Normalized Mean Squared Error (NMSE) and Normalized Cross-Correlation (NCC) metrics.
    '''
    def __init__(
            self,
            n_hidden_neurons: int,
            n_hidden_layers: int,
            n_collocation_points: int,
            temporal_frequency: float,
            region_size: np.ndarray,
            lambda_physics: float,
            lr: float = 0.001,
            iterations: int = 5_001,
            epsilon: float = 0.1
    ):
        super().__init__()
        activation = nn.Tanh
        torch.manual_seed(42)
        self.n_coll_points = n_collocation_points
        self.temporal_frequency = temporal_frequency
        self.region_size = torch.tensor(region_size, dtype=torch.float32)
        self.lambda_physics = lambda_physics
        self.lr = lr
        self.iterations = iterations
        
        self.epsilon = epsilon
        self.x_coll = self._collocation_points()

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
        x = self.input_layer(input)
        x = self.hidden_layers(x)
        x = self.output_layer(x)

        return x

    def _collocation_points(self):
        '''
        Generates collocation points to solve the Helmholtz equation during training.
        The collocation points are uniformly distributed within the specified 2D region.

        Returns
        -------
        x_collocation: torch.Tensor
            Collocation points in the 2D region, shape (n_collocation_points, 2).
        '''
        x1_collocation = torch.Tensor(self.n_coll_points, 1).uniform_(-self.region_size[0]/2, self.region_size[0]/2)
        x2_collocation = torch.Tensor(self.n_coll_points, 1).uniform_(-self.region_size[1]/2, self.region_size[1]/2)
        x_collocation = torch.cat([x1_collocation, x2_collocation], axis=1).requires_grad_(True)

        return x_collocation

    def _compute_pde(self, 
            x: torch.Tensor, 
            u: torch.Tensor, 
            k: float
        ):
        '''
        Solves the homogeneous Helmholtz equation at the given collocation points.

        Parameters
        ----------
        x: torch.Tensor
            Collocation points in the 2D region, shape (n_collocation_points, 2).
        u: torch.Tensor
            Predicted sound field at the collocation points, shape (n_collocation_points, 1).
        k: float
            Wave number.
        
        Returns
        -------
        u_pde: torch.Tensor
            Sound pressure in the frequency domain computed by solving the Helmholtz equation at the collocation points, shape (n_collocation_points, 1).
        '''
        # Gradients for solving the PDE in 2D
        du_dx = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        d2u_dx1x1 = torch.autograd.grad(du_dx[:,0], x, grad_outputs=torch.ones_like(du_dx[:,0]), create_graph=True)[0][:,0].view(-1,1)
        d2u_dx2x2 = torch.autograd.grad(du_dx[:,1], x, grad_outputs=torch.ones_like(du_dx[:,1]), create_graph=True)[0][:,1].view(-1,1)

        # Homogeneous Helmholtz equation
        u_pde = d2u_dx1x1 + d2u_dx2x2 + (k ** 2) * u

        return u_pde


    def predict(
            self,
            x: torch.Tensor
    ):
        '''
        Predicts the sound field at given coordinates x using the trained model.
        
        Parameters
        ----------
        x: torch.Tensor
            2D Cartesian coordinates where the sound field is estimated, shape (N, 2), where N is the number of points.
        
        Returns
        -------
        predicted_sound_field: torch.Tensor
            Estimated sound field at the coordinates x, shape (N, 1).
        '''

        predicted_sound_field = self(x)
        return predicted_sound_field
    
    def train_model(
            self,
            coords_microphones: torch.Tensor,
            train_data: torch.Tensor
        ):
        '''
        Trains the PINN model using the provided microphone coordinates and corresponding (ground truth) pressure.
        
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
        loss_pde_values = []
        loss_data_values = []

        for epoch in range(self.iterations):
            self.optimiser.zero_grad()

            # Data loss term
            pred_sound_field_data = self.predict(coords_microphones)
            loss_data = self.loss_function(pred_sound_field_data, train_data.view(-1,1))
            loss_data_values.append(loss_data.data)


            # Physics loss term
            x_collocation = self._collocation_points()
            pred_sound_field_coll = self.predict(x_collocation)
            
            # Homogeneous right-hand side: zeros
            rhs = torch.zeros_like(pred_sound_field_coll)
            
            pde_sound_field = self._compute_pde(x_collocation, pred_sound_field_coll, 2 * torch.pi * self.temporal_frequency / 343)
            loss_physics = self.loss_function(pde_sound_field, rhs)
            loss_pde_values.append(loss_physics.data)

            # Combined training loss
            loss = loss_data + self.lambda_physics * loss_physics

            loss_values.append(loss.item()) 
            loss.backward()
            self.optimiser.step()
            
        return loss_values

    def evaluate(
            self,
            u_estimated: torch.Tensor,
            u_ground_truth: torch.Tensor
        ):
        '''
        Evaluates the performance of the trained model using Normalized Mean Squared Error (NMSE) and Normalized Cross-Correlation (NCC) metrics.
        
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