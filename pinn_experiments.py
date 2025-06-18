import torch
import numpy as np
from src.room import Room
from src.models.pinn import PINNModel
from src.utils import plot_sound_field
from pathlib import Path
import argparse
import torchinfo

def main(seed):
    # Set random seeds for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Parameters of the experiment
    L = np.array([5,4]) # Room size in meters
    region_size = np.array([2,2])   # Omega region size in meters
    region_origin = np.array([0.5, 0.5])    # Origin of the region (bottom left corner) in meters
    fs = 16000  # Sampling frequency in Hz
    s = np.array([3.2,1])   # Source position in meters
    N_t = 2048  # Length of the RIR in samples
    N_fft = 1024    # FFT size
    n_grid_points = 30  # Number of grid points in each dimension (the grid will be n_grid_points x n_grid_points)
    T60 = 0.4   # Reverberation time in seconds
    lambda_physics = 0.001  # Physics loss weight
    iterations = 5_001 # Number of training iterations
    root_path = Path('output')  # Root path for saving results

    freq_axis = np.linspace(0,fs/2, N_fft//2+1) 
    
    # Room simulation
    room = Room(
        sample_rate=fs,
        room_size=L,
        T60=T60,
        source_position=s,
        omega_size=region_size,
        omega_origin=region_origin,
        n_grid_points=n_grid_points,
        len_rir=N_t,
        N_fft=N_fft, 
        randomized_ism=False,
        ism_order=40
    )

    ## Experiment 1
    n_coll_points = 200 # Number of collocation points
    freq_bins = np.arange(5,70,5) # Frequency bins to evaluate the model

    H_full = room.rirs  # Full RIRs for the entire grid
    H_mics, mic_idxs = room.select_rirs(n_mics=50, seed=seed) # Select RIRs for a subset of microphones (50 microphones)

    # Results storage
    nmse_freq = []
    ncc_freq = []

    for bin in freq_bins:
        temporal_frequency = freq_axis[bin]
        test_sound_field_estimated = torch.zeros((n_grid_points ** 2, 2))

        for idx, component in enumerate(['real', 'imag']):
            # Path to save results
            current_path = root_path.joinpath('pinn', 'exp1', str(seed), str(int(temporal_frequency)), component)
            current_path.mkdir(parents=True, exist_ok=True)

            # Initialize the PINN model
            model = PINNModel(
                n_hidden_neurons = 64,
                n_hidden_layers = 2,
                n_collocation_points = n_coll_points,
                temporal_frequency=temporal_frequency,
                region_size=region_size,
                lambda_physics=lambda_physics,
                lr = 0.001,
                iterations = iterations
            )
            # Uncomment the following line to print the model summary
            # torchinfo.summary(model)

            # Train coordinates: positions of the microphones
            train_coordinates = torch.tensor(room.centered_grid_positions[mic_idxs], dtype=torch.float32, requires_grad=True)
            
            # Test coordinates: positions of all the grid points
            test_coordinates = torch.tensor(room.centered_grid_positions, dtype=torch.float32, requires_grad=False)
            
            if component == 'real':
                train_pressure = torch.tensor(np.real(H_mics[:,bin]), dtype=torch.float32, requires_grad=True)
                test_pressure_gt = torch.tensor(np.real(H_full[:, bin]), dtype=torch.float32)
            elif component == 'imag':
                train_pressure = torch.tensor(np.imag(H_mics[:,bin]), dtype=torch.float32, requires_grad=True)
                test_pressure_gt = torch.tensor(np.imag(H_full[:, bin]), dtype=torch.float32)
            
            # Train model
            loss_values = model.train_model(train_coordinates, train_pressure)

            # Predict sound field at all grid points
            test_component_estimated = model.predict(test_coordinates)
            test_sound_field_estimated[:, idx] = test_component_estimated.squeeze()

            # Save model and results
            torch.save(model, current_path.joinpath('ckpt.pt'))
            np.save(current_path.joinpath('pressure_estimated.npy'), test_component_estimated.detach().numpy())
            np.save(current_path.joinpath('pressure_GT.npy'), test_pressure_gt.detach().numpy())
            
            nmse, ncc = model.evaluate(test_component_estimated, test_pressure_gt)
            plot_sound_field(n_grid_points, test_component_estimated.detach().numpy(), test_pressure_gt.detach().numpy(), 10 * torch.log10(torch.sum(nmse)).item(), current_path)
            

        # Combine real and imaginary components to form complex pressure
        test_pressure_estimated = test_sound_field_estimated[:,0] + 1j * test_sound_field_estimated[:,1]

        # Compute metrics
        nmse, ncc = model.evaluate(test_pressure_estimated, torch.tensor(H_full[:, bin]))

        nmse_freq.append(10 * torch.log10(torch.sum(nmse)).item())
        ncc_freq.append(ncc.item())
        
        print(f'Frequency: {freq_axis[bin]} - nmse: {nmse_freq[-1]:.2f} - ncc: {ncc_freq[-1]:.2f}')
        break

    np.savetxt(root_path.joinpath('pinn', 'exp1', str(seed), 'nmse.csv'), np.array(nmse_freq))
    np.savetxt(root_path.joinpath('pinn', 'exp1', str(seed), 'ncc.csv'), np.array(ncc_freq))


    ## Experiment 2
    n_coll_points = 200 # Number of collocation points
    bin = 25    # Frequency bin to evaluate the model
    temporal_frequency = freq_axis[bin]
    
    n_mics = [10,15,20,25,30,35,40,45,50] # Number of microphones M

    H_full = room.rirs  # Full RIRs for the entire grid

    # Results storage
    nmse_mic = []
    ncc_mic = []

    for M in n_mics:
        H_mics, mic_idxs = room.select_rirs(n_mics=M, seed=seed)    # Training RIRs for a subset of microphones (M microphones)
        
        test_sound_field_estimated = torch.zeros((n_grid_points ** 2, 2))

        for idx, component in enumerate(['real', 'imag']):
            # Path to save results
            current_path = root_path.joinpath('pinn', 'exp2', str(seed), str(M), component)
            current_path.mkdir(parents=True, exist_ok=True)

            # Initialize the PINN model
            model = PINNModel(
                n_hidden_neurons = 64,
                n_hidden_layers = 2,
                n_collocation_points = n_coll_points,
                temporal_frequency=temporal_frequency,
                region_size=region_size,
                lambda_physics=lambda_physics,
                lr = 0.001,
                iterations = iterations
            )

            # Train coordinates: positions of the microphones
            train_coordinates = torch.tensor(room.centered_grid_positions[mic_idxs], dtype=torch.float32, requires_grad=True)
            # Test coordinates: positions of all the grid points
            test_coordinates = torch.tensor(room.centered_grid_positions, dtype=torch.float32, requires_grad=False)
            
            if component == 'real':
                train_pressure = torch.tensor(np.real(H_mics[:,bin]), dtype=torch.float32, requires_grad=True)
                test_pressure_gt = torch.tensor(np.real(H_full[:, bin]), dtype=torch.float32)
            elif component == 'imag':
                train_pressure = torch.tensor(np.imag(H_mics[:,bin]), dtype=torch.float32, requires_grad=True)
                test_pressure_gt = torch.tensor(np.imag(H_full[:, bin]), dtype=torch.float32)
            
            # Train model
            loss_values = model.train_model(train_coordinates, train_pressure)

            # Predict sound field at all grid points
            test_component_estimated = model.predict(test_coordinates)
            test_sound_field_estimated[:, idx] = test_component_estimated.squeeze()

            # Save model and results
            torch.save(model, current_path.joinpath('ckpt.pt'))
            np.save(current_path.joinpath('pressure_estimated.npy'), test_component_estimated.detach().numpy())
            np.save(current_path.joinpath('pressure_GT.npy'), test_pressure_gt.detach().numpy())
            
            nmse, ncc = model.evaluate(test_component_estimated, test_pressure_gt)
            plot_sound_field(n_grid_points, test_component_estimated.detach().numpy(), test_pressure_gt.detach().numpy(), 10 * torch.log10(torch.sum(nmse)).item(), current_path)

        # Combine real and imaginary components to form complex pressure
        test_pressure_estimated = test_sound_field_estimated[:,0] + 1j * test_sound_field_estimated[:,1]

        # Compute metrics
        nmse, ncc = model.evaluate(test_pressure_estimated, torch.tensor(H_full[:, bin]))

        nmse_mic.append(10 * torch.log10(torch.sum(nmse)).item())
        ncc_mic.append(ncc.item())
        
        print(f'Mics: {M} - nmse: {nmse_mic[-1]:.2f} - ncc: {ncc_mic[-1]:.2f}')

    np.savetxt(root_path.joinpath('pinn', 'exp2', str(seed), 'nmse.csv'), np.array(nmse_mic))
    np.savetxt(root_path.joinpath('pinn', 'exp2', str(seed), 'ncc.csv'), np.array(ncc_mic))


    ## Experiment 3
    collocation_points = [52,100,152,200,252,300,352,400,452,500,552,600]   # Number of collocation points (52, 152, 252... are chosen to be divisible by 4)
    bin = 25    # Frequency bin to evaluate the model
    temporal_frequency = freq_axis[bin]

    H_full = room.rirs  # Full RIRs for the entire grid
    H_mics, mic_idxs = room.select_rirs(n_mics=50, seed=seed)   # Select RIRs for a subset of microphones (50 microphones)

    # Results storage
    nmse_coll = []
    ncc_coll = []

    for N_coll in collocation_points:
        
        test_sound_field_estimated = torch.zeros((n_grid_points ** 2, 2))

        for idx, component in enumerate(['real', 'imag']):
            # Path to save results
            current_path = root_path.joinpath('pinn', 'exp3', str(seed), str(N_coll), component)
            current_path.mkdir(parents=True, exist_ok=True)

            # Initialize the PINN model
            model = PINNModel(
                n_hidden_neurons = 64,
                n_hidden_layers = 2,
                n_collocation_points = N_coll,
                temporal_frequency=temporal_frequency,
                region_size=region_size,
                lambda_physics=lambda_physics,
                lr = 0.001,
                iterations = iterations
            )

            # Train coordinates: positions of the microphones
            train_coordinates = torch.tensor(room.centered_grid_positions[mic_idxs], dtype=torch.float32, requires_grad=True)
            # Test coordinates: positions of all the grid points
            test_coordinates = torch.tensor(room.centered_grid_positions, dtype=torch.float32, requires_grad=False)
            
            if component == 'real':
                train_pressure = torch.tensor(np.real(H_mics[:,bin]), dtype=torch.float32, requires_grad=True)
                test_pressure_gt = torch.tensor(np.real(H_full[:, bin]), dtype=torch.float32)
            elif component == 'imag':
                train_pressure = torch.tensor(np.imag(H_mics[:,bin]), dtype=torch.float32, requires_grad=True)
                test_pressure_gt = torch.tensor(np.imag(H_full[:, bin]), dtype=torch.float32)
            
            # Train model
            loss_values = model.train_model(train_coordinates, train_pressure)

            # Predict sound field at all grid points
            test_component_estimated = model.predict(test_coordinates)
            test_sound_field_estimated[:, idx] = test_component_estimated.squeeze()

            # Save model and results
            torch.save(model, current_path.joinpath('ckpt.pt'))
            np.save(current_path.joinpath('pressure_estimated.npy'), test_component_estimated.detach().numpy())
            np.save(current_path.joinpath('pressure_GT.npy'), test_pressure_gt.detach().numpy())
            
            nmse, ncc = model.evaluate(test_component_estimated, test_pressure_gt)
            plot_sound_field(n_grid_points, test_component_estimated.detach().numpy(), test_pressure_gt.detach().numpy(), 10 * torch.log10(torch.sum(nmse)).item(), current_path)

        # Combine real and imaginary components to form complex pressure
        test_pressure_estimated = test_sound_field_estimated[:,0] + 1j * test_sound_field_estimated[:,1]

        # Compute metrics
        nmse, ncc = model.evaluate(test_pressure_estimated, torch.tensor(H_full[:, bin]))

        nmse_coll.append(10 * torch.log10(torch.sum(nmse)).item())
        ncc_coll.append(ncc.item())
        
        print(f'N_coll: {N_coll} - nmse: {nmse_coll[-1]:.2f} - ncc: {ncc_coll[-1]:.2f}')

    np.savetxt(root_path.joinpath('pinn', 'exp3', str(seed), 'nmse.csv'), np.array(nmse_coll))
    np.savetxt(root_path.joinpath('pinn', 'exp3', str(seed), 'ncc.csv'), np.array(ncc_coll))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--seed', type=int, default=42)

    args = parser.parse_args()
    seed = args.seed

    print(f'Current seed: {seed}')
    main(seed)