import numpy as np
from signal_models import model
from data.load_data import load_grad
from dmipy.core.acquisition_scheme import acquisition_scheme_from_bvalues


def simulate_training_signals_dmipy(model, grad_file, nvox):
 
    grad = load_grad(grad_file)
    #convert to dmipy acquisition scheme
    acq_scheme = acquisition_scheme_from_bvalues(grad[:,0]*1e-9, grad[:,1:3])

        
    # simulates training signals for a given dmipy model and acquisition scheme
    # Returns training signals in "unscaled" format - i.e. close to 1, diffusivitys
    # for example are in μm^2/ms
    # model, acq_scheme are in dmipy model dictionary 

    nparam = sum(model.parameter_cardinality.values())
    ncomp = len(model.partial_volume_names)

    ground_truth_parameter_vector = np.zeros((nvox, nparam))
    scaled_ground_truth_parameter_vector = np.zeros((nvox, nparam))

    j = 0
    for param in model.parameter_names:
        # simulate the ground truth parameter values
        if model.parameter_cardinality[param] == 2:
            param_0_min = model.parameter_ranges[param][0][0]
            param_0_max = model.parameter_ranges[param][0][1]
            param_1_min = model.parameter_ranges[param][1][0]
            param_1_max = model.parameter_ranges[param][1][1]

            param_0_scale = model.parameter_scales[param][0]
            param_1_scale = model.parameter_scales[param][1]

            ground_truth_parameter_vector[:, j] = np.random.uniform(param_0_min, param_0_max, (1, nvox))
            scaled_ground_truth_parameter_vector[:, j] = param_0_scale * np.random.uniform(param_0_min, param_0_max, (1, nvox))
            j = j + 1
            ground_truth_parameter_vector[:, j] = np.random.uniform(param_1_min, param_1_max, (1, nvox))
            scaled_ground_truth_parameter_vector[:, j] = param_1_scale * np.random.uniform(param_1_min, param_1_max, (1, nvox))
            j = j + 1
        elif (j == nparam - 1) & (ncomp > 1):
            # this is the final partial volume
            # makesure the partial volumes sum to 1
            if ncomp == 2:
                ground_truth_parameter_vector[:, j] = 1 - ground_truth_parameter_vector[:, j-1]
                scaled_ground_truth_parameter_vector[:, j] = ground_truth_parameter_vector[:, j]
            else:
                ground_truth_parameter_vector[:, j] = 1 - np.sum(ground_truth_parameter_vector[:, (j-(ncomp-1)):(j-1)], axis=1)
                scaled_ground_truth_parameter_vector[:, j] = ground_truth_parameter_vector[:, j]
        else:
            param_min = model.parameter_ranges[param][0]
            param_max = model.parameter_ranges[param][1]

            param_scale = model.parameter_scales[param]

            ground_truth_parameter_vector[:, j] = np.random.uniform(param_min, param_max, (1, nvox))
            scaled_ground_truth_parameter_vector[:, j] = param_scale * ground_truth_parameter_vector[:, j]
            j = j + 1

    # convert the ground truth parameter values into 

    training_signals = model.simulate_signal(acq_scheme, scaled_ground_truth_parameter_vector)
    training_signals = add_noise(training_signals)

    return training_signals, ground_truth_parameter_vector, model, acq_scheme


def add_noise(signal,scale=0.02):
    signal_real = signal + np.random.normal(scale=scale, size=np.shape(signal))
    signal_imag = np.random.normal(scale=scale, size=np.shape(signal))
    noisy_signal = np.sqrt(signal_real**2 + signal_imag**2)

    return noisy_signal


def simulate_training_signals(model, grad_file, nvox):
    #simulate training signals using the torch functions
    grad = load_grad(grad_file)

        
    # simulates training signals for a given dmipy model and acquisition scheme
    # Returns training signals in "unscaled" format - i.e. close to 1, diffusivitys
    # for example are in μm^2/ms
    # model, acq_scheme are in dmipy model dictionary 

        
    

    training_signals = model.simulate_signal(acq_scheme, scaled_ground_truth_parameter_vector)
    training_signals = add_noise(training_signals)

    return training_signals, ground_truth_parameter_vector