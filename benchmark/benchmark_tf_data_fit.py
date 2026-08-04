import os
# Force TensorFlow to use CPU only to avoid CUDA/PTX compilation errors
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import numpy as np 
import tensorflow as tf 
import csv
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# Disable TF oneDNN verbose logs
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


def Translate_Tensorflow_MLP(file_out: str, input_names: list[str], output_names: list[str], model: tf.keras.models.Sequential, \
                             scaler_input: str = "minmax", input_norm_1: list[float] = [], input_norm_2: list[float] = [], \
                             scaler_output: str = "minmax", output_norm_1: list[float] = [], output_norm_2: list[float] = []):
    # MLP config
    model_config = model.get_config()

    n_inputs = len(input_names)
    n_outputs = len(output_names)

    if len(input_norm_2) != len(input_norm_1):
        raise Exception("Upper and lower input normalizations should have the same length")
    if len(output_norm_2) != len(output_norm_1):
        raise Exception("Upper and lower output normalizations should have the same length")
    if len(input_norm_2) > 0 and len(input_norm_1) != n_inputs:
        raise Exception("Input normalization not provided for all inputs")
    if len(output_norm_2) > 0 and len(output_norm_1) != n_outputs:
        raise Exception("Output normalization not provided for all outputs")

    fid = open(file_out + '.mlp', 'w+')
    fid.write("<header>\n\n")
    n_layers = len(model_config['layers'])

    fid.write('[number of layers]\n%i\n\n' % n_layers)
    fid.write('[neurons per layer]\n')
    activation_functions = []

    for iLayer in range(n_layers-1):
        layer_class = model_config['layers'][iLayer]['class_name']
        if layer_class == 'InputLayer':
            activation_functions.append('linear')
            n_neurons = model_config['layers'][iLayer]['config']['batch_shape'][1]
        else:
            activation_functions.append(model_config['layers'][iLayer]['config']['activation'])
            n_neurons = model_config['layers'][iLayer]['config']['units']
        fid.write('%i\n' % n_neurons)
    fid.write('%i\n' % n_outputs)

    activation_functions.append('linear')

    fid.write('\n[activation function]\n')
    for iLayer in range(n_layers):
        fid.write(activation_functions[iLayer] + '\n')

    fid.write('\n[input names]\n')
    for input in input_names:
        fid.write(input + '\n')
    
    fid.write("\n[input regularization method]\n%s\n" % scaler_input)

    if len(input_norm_1) > 0:
        fid.write('\n[input normalization]\n')
        for i in range(len(input_names)):
            fid.write('%+.16e\t%+.16e\n' % (input_norm_1[i], input_norm_2[i]))
    
    fid.write('\n[output names]\n')
    for output in output_names:
        fid.write(output+'\n')
    
    fid.write("\n[output regularization method]\n%s\n" % scaler_output)

    if len(output_norm_1) > 0:
        fid.write('\n[output normalization]\n')
        for i in range(len(output_names)):
            fid.write('%+.16e\t%+.16e\n' % (output_norm_1[i], output_norm_2[i]))

    fid.write("\n</header>\n")
    fid.write('\n[weights per layer]\n')
    for layer in model.layers:
        fid.write('<layer>\n')
        weights = layer.get_weights()[0]
        for row in weights:
            fid.write("\t".join(f'{w:+.16e}' for w in row) + "\n")
        fid.write('</layer>\n')
    
    fid.write('\n[biases per layer]\n')
    fid.write('%+.16e\t%+.16e\t%+.16e\n' % (0.0, 0.0, 0.0))

    for layer in model.layers:
        biases = layer.get_weights()[1]
        fid.write("\t".join([f'{b:+.16e}' for b in biases]) + "\n")

    fid.close()

# ============================================================================
# 1. Generate Dataset
# ============================================================================
t = np.linspace(0, 1, 5000)
u = (1 - t) * np.cos(48*np.pi*t)
v = (1 - t) * np.sin(48*np.pi*t)
y = np.sin(2*np.pi*(u*u + v))

X_dim = np.column_stack([u, v])
Y_dim = y

with open("reference_data.csv", "w") as fid:
    fid.write("u\tv\ty\n")
    np.savetxt(fid, np.column_stack([X_dim, Y_dim]), delimiter="\t", fmt="%.16e")

# ============================================================================
# 2. Normalize Data
# ============================================================================
scaler_function_x = "minmax"
scaler_function_y = "minmax"

scaler_x = MinMaxScaler()
scaler_y = MinMaxScaler()

scaler_x.fit(X_dim)
scaler_y.fit(Y_dim[:, np.newaxis])

X_norm = scaler_x.transform(X_dim)
Y_norm = scaler_y.transform(Y_dim[:, np.newaxis])

input_norm_1 = scaler_x.data_min_ 
input_norm_2 = scaler_x.data_max_
output_norm_1 = scaler_y.data_min_ 
output_norm_2 = scaler_y.data_max_

# ============================================================================
# 3. Build Model
# ============================================================================
hidden_layers = [16, 16]

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Input([2]))
for NN in hidden_layers:
    model.add(tf.keras.layers.Dense(NN, activation='tanh', kernel_initializer="he_uniform"))
model.add(tf.keras.layers.Dense(1, activation='linear'))

# Save initial weights so C++ can start from the exact same point
Translate_Tensorflow_MLP(file_out="initial_model", input_names=["u", "v"], output_names=["y"], model=model,
                         scaler_input=scaler_function_x, input_norm_1=input_norm_1, input_norm_2=input_norm_2,
                         scaler_output=scaler_function_y, output_norm_1=output_norm_1, output_norm_2=output_norm_2)

# ============================================================================
# 4. Train
# ============================================================================
opt = tf.keras.optimizers.Adam(learning_rate=1e-3, beta_1=0.9, beta_2=0.999, epsilon=1e-8)
model.compile(optimizer=opt, loss="mean_squared_error", metrics=["mape"])

history = model.fit(X_norm, Y_norm, epochs=200, batch_size=32, shuffle=True, verbose=0)

# ============================================================================
# 5. Evaluate and Print Final MSE
# ============================================================================
Y_norm_pred = model.predict(X_norm, verbose=0)
Y_dim_pred = scaler_y.inverse_transform(Y_norm_pred)
mse_raw = np.mean((Y_dim_pred[:, 0] - Y_dim)**2)
print(f"\n======================================")
print(f"Final MSE (raw y space): {mse_raw:.10e}")
print(f"======================================\n")

# Optional: Save final model if you want to test your C++ inference tool separately later
Translate_Tensorflow_MLP(file_out="MLP_test", input_names=["u", "v"], output_names=["y"], model=model,
                         scaler_input=scaler_function_x, input_norm_1=input_norm_1, input_norm_2=input_norm_2,
                         scaler_output=scaler_function_y, output_norm_1=output_norm_1, output_norm_2=output_norm_2)

# ============================================================================
# 6. Visualize Convergence Comparison
# ============================================================================
# Convert TF normalized loss to raw y-space loss to match C++ output
# MSE_raw = MSE_norm * (y_max - y_min)^2
y_range = output_norm_2[0] - output_norm_1[0]
tf_loss_raw = [loss * (y_range**2) for loss in history.history['loss']]
tf_epochs = np.arange(1, len(tf_loss_raw) + 1)

# Check if C++ history file exists
cpp_file = "data_fitting_history.csv"
try:
    cpp_data = np.genfromtxt(cpp_file, delimiter=',', names=True)
    cpp_epochs = cpp_data['epoch']
    cpp_loss_data = cpp_data['loss_data']
    
    plt.figure(figsize=(10, 6))
    plt.semilogy(tf_epochs, tf_loss_raw, 'r-', label='TensorFlow Data Loss (Raw Space)', linewidth=2)
    plt.semilogy(cpp_epochs, cpp_loss_data, 'b--', label='MLPCpp Data Loss (Raw Space)', linewidth=2)
    plt.title('Data Fitting Convergence: TensorFlow vs C++')
    plt.xlabel('Epoch')
    plt.ylabel('Mean Squared Error (log scale)')
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig('results/convergence_comparison.png', dpi=300)
    print("Convergence plot saved to convergence_comparison.png")
    
except Exception as e:
    print(f"Could not read {cpp_file} or plot. Make sure to run the C++ test case first. Error: {e}")