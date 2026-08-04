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

# Match C++ double precision
tf.keras.backend.set_floatx('float64')

def Translate_Tensorflow_MLP(file_out: str, input_names: list[str], output_names: list[str], model: tf.keras.models.Sequential, \
                             scaler_input: str = "minmax", input_norm_1: list[float] = [], input_norm_2: list[float] = [], \
                             scaler_output: str = "minmax", output_norm_1: list[float] = [], output_norm_2: list[float] = []):
    model_config = model.get_config()
    n_inputs = len(input_names)
    n_outputs = len(output_names)

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

input_norm_1 = scaler_x.data_min_ 
input_norm_2 = scaler_x.data_max_
output_norm_1 = scaler_y.data_min_
output_norm_2 = scaler_y.data_max_

# ============================================================================
# 3. Build Model 
# ============================================================================
hidden_layers = [16, 16]

# Match C++ RandomWeights() which uses uniform [-1.0, 1.0] for BOTH weights and biases
init_weights = tf.keras.initializers.RandomUniform(minval=-1.0, maxval=1.0)
init_biases = tf.keras.initializers.RandomUniform(minval=-1.0, maxval=1.0)

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Input([2]))
for NN in hidden_layers:
    model.add(tf.keras.layers.Dense(NN, activation='tanh', kernel_initializer=init_weights, bias_initializer=init_biases))
model.add(tf.keras.layers.Dense(1, activation='linear', kernel_initializer=init_weights, bias_initializer=init_biases))

# ============================================================================
# 4. Physics-informed training loop
# ============================================================================
x_min_tf = tf.constant(input_norm_1, dtype=tf.float64)
x_max_tf = tf.constant(input_norm_2, dtype=tf.float64)
y_min_tf = tf.constant(output_norm_1[0], dtype=tf.float64)
y_max_tf = tf.constant(output_norm_2[0], dtype=tf.float64)

def normalize_x(x_raw):
    return (x_raw - x_min_tf) / (x_max_tf - x_min_tf)

def denormalize_y(y_norm):
    return y_norm * (y_max_tf - y_min_tf) + y_min_tf

Ncoll = 200
t_coll = np.linspace(0.0, 1.0, Ncoll)
u_coll = u.min() + t_coll * (u.max() - u.min())
v_coll = v.min() + t_coll * (v.max() - v.min())
X_coll_raw = tf.constant(np.column_stack([u_coll, v_coll]), dtype=tf.float64)

# Match C++: physics_batch_size = 32
PHYSICS_BATCH_SIZE = 32 
coll_perm = np.arange(Ncoll)
coll_idx = 0

def next_physics_batch():
    global coll_perm, coll_idx
    if coll_idx == 0:
        np.random.shuffle(coll_perm)
    batch_idx = coll_perm[coll_idx:coll_idx+PHYSICS_BATCH_SIZE]
    coll_idx += PHYSICS_BATCH_SIZE
    if coll_idx >= Ncoll:
        coll_idx = 0
    return tf.gather(X_coll_raw, batch_idx)

LAMBDA_PHYS = 1.0

def physics_residual_loss(model, x_coll_raw):
    u_raw = x_coll_raw[:, 0:1]
    v_raw = x_coll_raw[:, 1:2]
    with tf.GradientTape() as inner_tape:
        inner_tape.watch(u_raw)
        x_raw = tf.concat([u_raw, v_raw], axis=1)
        x_norm = normalize_x(x_raw)
        y_norm = model(x_norm, training=True)
        y_raw = denormalize_y(y_norm)
    dy_du = inner_tape.gradient(y_raw, u_raw)
    return tf.reduce_mean(tf.square(dy_du))

X_raw_tf = tf.constant(X_dim, dtype=tf.float64)
Y_raw_tf = tf.constant(Y_dim[:, np.newaxis], dtype=tf.float64)

BATCH_SIZE = 32
EPOCHS = 200
N_train = X_raw_tf.shape[0]
steps_per_epoch = (N_train + BATCH_SIZE - 1) // BATCH_SIZE

opt = tf.keras.optimizers.Adam(learning_rate=1e-3, beta_1=0.9, beta_2=0.999, epsilon=1e-8)

print(f"\nStarting training ({EPOCHS} epochs)...")
print(f"Physics mini-batch size: {PHYSICS_BATCH_SIZE}\n")
print(f"{'Epoch':<8}{'L_data':<16}{'L_phys':<16}{'lambda':<12}{'L_total':<16}")
print("-" * 68)

# Lists to store TF history for plotting
tf_epochs = []
tf_data_loss_hist = []
tf_phys_loss_hist = []
tf_total_loss_hist = []

rng = np.random.default_rng(0)
for epoch in range(EPOCHS):
    perm = rng.permutation(N_train)
    epoch_data_loss  = 0.0
    epoch_phys_loss  = 0.0
    epoch_total_loss = 0.0

    for b in range(steps_per_epoch):
        idx = perm[b * BATCH_SIZE: (b + 1) * BATCH_SIZE]
        xb_raw = tf.gather(X_raw_tf, idx)
        yb_raw = tf.gather(Y_raw_tf, idx)

        # Match C++: get 32 physics points for this step
        x_coll_batch = next_physics_batch()

        with tf.GradientTape() as outer_tape:
            xb_norm = normalize_x(xb_raw)
            yb_norm_pred = model(xb_norm, training=True)
            yb_raw_pred = denormalize_y(yb_norm_pred)
            data_loss = tf.reduce_mean(tf.square(yb_raw_pred - yb_raw))

            phys_loss = physics_residual_loss(model, x_coll_batch)

            total_loss = data_loss + LAMBDA_PHYS * phys_loss

        grads = outer_tape.gradient(total_loss, model.trainable_variables)
        opt.apply_gradients(zip(grads, model.trainable_variables))

        epoch_data_loss  += float(data_loss)
        epoch_phys_loss  += float(phys_loss)
        epoch_total_loss += float(total_loss)

    epoch_data_loss  /= steps_per_epoch
    epoch_phys_loss  /= steps_per_epoch
    epoch_total_loss /= steps_per_epoch
    
    # Save to history lists
    tf_epochs.append(epoch + 1)
    tf_data_loss_hist.append(epoch_data_loss)
    tf_phys_loss_hist.append(epoch_phys_loss)
    tf_total_loss_hist.append(epoch_total_loss)

    if epoch % 10 == 0 or epoch == EPOCHS - 1:
        print(f"{epoch + 1:<8}{epoch_data_loss:<16.4e}{epoch_phys_loss:<16.4e}"
              f"{LAMBDA_PHYS:<12.4f}{epoch_total_loss:<16.4e}")
        
# ============================================================================
# 5. Evaluate and Print Final Loss (raw y-space, matches C++ reporting)
# ============================================================================
X_norm_full = normalize_x(X_raw_tf)
Y_norm_pred_full = model(X_norm_full, training=False)
Y_raw_pred_full = denormalize_y(Y_norm_pred_full).numpy()
mse_raw = np.mean((Y_raw_pred_full[:, 0] - Y_dim) ** 2)

phys_final = float(physics_residual_loss(model, X_coll_raw))

print(f"\n======================================")
print(f"Final data loss    (raw y space): {mse_raw:.10e}")
print(f"Final physics loss:               {phys_final:.10e}")
print(f"Final total loss:                 {mse_raw + LAMBDA_PHYS * phys_final:.10e}")
print(f"======================================\n")

# Optional: Save final model to test C++ inference tool separately
Translate_Tensorflow_MLP(file_out="MLP_test", input_names=["u", "v"], output_names=["y"], model=model,
                         scaler_input=scaler_function_x, input_norm_1=input_norm_1, input_norm_2=input_norm_2,
                         scaler_output=scaler_function_y, output_norm_1=output_norm_1, output_norm_2=output_norm_2)


# ============================================================================
# 6. Visualize Convergence Comparison
# ============================================================================
print("--- Generating Convergence Plots ---")
cpp_file = "pinn_training_history.csv"
try:
    cpp_data = np.genfromtxt(cpp_file, delimiter=',', names=True)
    cpp_epochs = cpp_data['epoch']
    cpp_loss_data = cpp_data['loss_data']
    cpp_loss_phys = cpp_data['loss_phys']
    cpp_lambda = cpp_data['lambda']
    cpp_loss_total = cpp_data['loss_total']
    
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Data Loss
    axs[0, 0].semilogy(tf_epochs, tf_data_loss_hist, 'r-', label='TF Data Loss', linewidth=2)
    axs[0, 0].semilogy(cpp_epochs, cpp_loss_data, 'b--', label='MLPCpp Data Loss', linewidth=2)
    axs[0, 0].set_title('Data Fitting Loss')
    axs[0, 0].set_xlabel('Epoch')
    axs[0, 0].set_ylabel('Loss (log scale)')
    axs[0, 0].legend()
    axs[0, 0].grid(True, which="both", ls="--", alpha=0.5)
    
    # Plot 2: Physics Loss
    axs[0, 1].semilogy(tf_epochs, tf_phys_loss_hist, 'r-', label='TF Physics Loss', linewidth=2)
    axs[0, 1].semilogy(cpp_epochs, cpp_loss_phys, 'b--', label='MLPCpp Physics Loss', linewidth=2)
    axs[0, 1].set_title('Physics Residual Loss')
    axs[0, 1].set_xlabel('Epoch')
    axs[0, 1].set_ylabel('Loss (log scale)')
    axs[0, 1].legend()
    axs[0, 1].grid(True, which="both", ls="--", alpha=0.5)
    
    # Plot 3: Total Loss
    axs[1, 0].semilogy(tf_epochs, tf_total_loss_hist, 'r-', label='TF Total Loss', linewidth=2)
    axs[1, 0].semilogy(cpp_epochs, cpp_loss_total, 'b--', label='MLPCpp Total Loss', linewidth=2)
    axs[1, 0].set_title('Total Loss')
    axs[1, 0].set_xlabel('Epoch')
    axs[1, 0].set_ylabel('Loss (log scale)')
    axs[1, 0].legend()
    axs[1, 0].grid(True, which="both", ls="--", alpha=0.5)
    
    # Plot 4: Lambda
    axs[1, 1].semilogy(cpp_epochs, cpp_lambda, 'b--', label='MLPCpp Lambda', linewidth=2)
    axs[1, 1].axhline(y=LAMBDA_PHYS, color='r', linestyle='-', label='TF Lambda (Fixed)')
    axs[1, 1].set_title('Annealing Parameter Lambda')
    axs[1, 1].set_xlabel('Epoch')
    axs[1, 1].set_ylabel('Lambda (log scale)')
    axs[1, 1].legend()
    axs[1, 1].grid(True, which="both", ls="--", alpha=0.5)
    
    plt.tight_layout()
    plt.savefig('pinn_convergence_comparison.png', dpi=300)
    print("Plot saved to pinn_convergence_comparison.png")
    plt.show()
    
except Exception as e:
    print(f"Could not read {cpp_file} or plot. Run the C++ test case first. Error: {e}")