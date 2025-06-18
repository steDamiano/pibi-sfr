import numpy as np
import pyroomacoustics as pra
import matplotlib.pyplot as plt

class Room:
    '''
    Class to generate Room Impulse Responses (RIRs) for a given room configuration. Only 2D rooms are supported.

    Attributes
    ----------
    sample_rate: int
        Sample rate of the generated RIRs.
    room_size: np.ndarray
        Size of the room in meters, given as a 2D array [width, height].
    T60: float
        Reverberation time (T60) of the room in seconds.
    source_position: np.ndarray 
        Position of the sound source in the room, given as a 2D array [x, y].
    omega_size: np.ndarray
        Size of the grid in meters, given as a 2D array [width, height].
    omega_origin: np.ndarray
        Origin of the grid in meters, given as a 2D array [x, y].
    n_grid_points: int
        Number of grid points along each dimension of the grid.
    len_rir: int, optional
        Length of the RIRs in samples. Default is 2048.
    N_fft: int, optional
        Number of FFT points for computing the frequency-domain RIRs. Default is 1024.
    randomized_ism: bool, optional
        Whether to use randomized image source model. Default is False.
    ism_order: int, optional
        Order of the image source model. Default is 40.
    
    Methods
    -------
    select_rirs(n_mics, seed):
        Selects a random subset of RIRs based on the number of microphones specified, using the specified seed.
    
    '''
    def __init__(
            self,
            sample_rate: int,
            room_size: np.ndarray,
            T60: float,
            source_position: np.ndarray,
            omega_size: np.ndarray,
            omega_origin: np.ndarray,
            n_grid_points: int,
            len_rir: int = 2048,
            N_fft: int = 1024,
            randomized_ism: bool = False,
            ism_order: int = 40,
        ):
        self.sample_rate = sample_rate
        self.room_size = room_size
        self.T60 = T60
        self.source_position = source_position
        
        # Optional parameters
        self.len_rir = len_rir
        self.N_fft = N_fft
        self.randomized_ism = randomized_ism
        self.ism_order = ism_order

        self.n_grid_points = n_grid_points
        self.omega_size = omega_size
        self.omega_origin = omega_origin

        self.grid_positions = self._generate_omega_grid(
            n_grid_points = n_grid_points,
            omega_size = omega_size,
            omega_origin = omega_origin
        )

        self.centered_grid_positions = self.grid_positions - self.omega_size/2 - self.omega_origin
        self.centered_source_position = self.source_position - self.omega_size/2 - self.omega_origin

        self.rirs = self._generate_rirs()
    
    def _generate_omega_grid(
            self,
            n_grid_points: int,
            omega_size: float,
            omega_origin: np.ndarray
        ):
        '''
        Generates a grid of points in the chosen 2D region of space Omega.

        Parameters
        ----------
        n_grid_points: int
            Number of grid points along each dimension.
        omega_size: np.ndarray
            Size of the grid in meters, given as a 2D array [width, height].
        omega_origin: np.ndarray
            Origin of the grid in meters, given as a 2D array [x, y].
        
        Returns
        -------
        omega_grid: np.ndarray
            Cartesian coordinates of the grid points, shape (n_grid_points^2, 2).
        '''
        xx, yy = np.meshgrid(np.linspace(0, omega_size[0], n_grid_points) + omega_origin[0], np.linspace(0, omega_size[1], n_grid_points) + omega_origin[1])
        
        omega_grid = np.concatenate((np.expand_dims(xx.ravel(),axis=1), np.expand_dims(yy.ravel(),axis=1)), axis=1)
        
        return omega_grid
    
    def _generate_rirs(
            self,
            ) -> np.ndarray:
        '''
        Generates Room Impulse Responses (RIRs) in the frequency domain for the given room configuration.

        Returns
        -------
        H: np.ndarray
            Frequency-domain representation of the RIRs, shape (n_grid_points^2, N_fft).
        '''
        e_absorption, _ = pra.inverse_sabine(self.T60, self.room_size)
        room = pra.ShoeBox(self.room_size, fs = self.sample_rate, materials = pra.Material(e_absorption), max_order = self.ism_order, use_rand_ism = self.randomized_ism, max_rand_disp = 0.08)
        room.add_source(self.source_position)
        room.add_microphone_array(self.grid_positions.T)
        room.compute_rir()

        H = np.zeros((len(self.grid_positions), self.N_fft), dtype=np.complex64)
        for m in range(len(self.grid_positions)):
            h = room.rir[m][0][:self.len_rir]
            h /= np.linalg.norm(h)
            H[m,:] = np.fft.fft(h, n=self.N_fft)
        
        return H
    
    def select_rirs(
            self,
            n_mics: int,
            seed: int = 42
    ):
        '''
        Selects a random subset of RIRs based on the number of microphones specified.

        Parameters
        ----------
        n_mics: int
            Number of RIRs to select.
        seed: int, optional
            Random seed for reproducibility. Default is 42.
        Returns
        -------
        H_microphones: np.ndarray
            Frequency-domain representation of the selected RIRs, shape (n_mics, N_fft).
        meas_mics_idxs: np.ndarray
            Indices of the selected microphones in the original grid.
        '''
        np.random.seed(seed)

        meas_mics_idxs = np.sort(np.random.choice(len(self.grid_positions), size=n_mics, replace=False))

        mask = np.zeros(self.n_grid_points ** 2)
        mask[meas_mics_idxs] = 1
        
        H_microphones = self.rirs[meas_mics_idxs]

        return H_microphones, np.sort(meas_mics_idxs)