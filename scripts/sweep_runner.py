"""
sweep_runner.py
===============
Trains the full Poirazi 3-stage sweep (constrained + all-to-all baselines),
saves each model to a structured directory, and produces a pandas DataFrame
summarising every run.

Directory layout
────────────────
sweep_results/
    runs/
        constrained_s128_d4_so128/
            model.keras          ← saved Keras model
            history.json         ← per-epoch loss/MAE
            config.json          ← hyperparameters + mask summary
        alltoall_s128_d4_so128/
            ...
    sweep_summary.csv            ← one row per run, all metrics

Usage
─────
    python sweep_runner.py

    # Or import and call from a notebook:
    from sweep_runner import run_sweep
    df = run_sweep(epochs=200, n_samples=5000)
"""

import os
import json
import time
import warnings
import numpy as np
import pandas as pd
import tensorflow as tf
import keras
import matplotlib.pyplot as plt

from bio_masks import sweep_masks

from visualize_masks import plot_masks, plot_architectures, plot_all, verify_figure_masks, print_verification



# Callback to compute per-target validation MAE at end of each epoch
class PerTargetMAECallback(tf.keras.callbacks.Callback):
    def __init__(self, x_val, y_val, x_train=None, y_train=None):
        super().__init__()
        self.x_val = x_val
        self.y_val = y_val
        self.x_train = x_train
        self.y_train = y_train
        # Record both validation and training per-target MAE per epoch
        self.history = {
            "val_mae_az": [], "val_mae_el": [],
            "mae_az": [], "mae_el": [],
        }

    def on_epoch_end(self, epoch, logs=None):
        
        y_pred_val = self.model.predict(self.x_val, batch_size=32, verbose=0)

        abs_err_val = np.abs(y_pred_val - self.y_val)
        val_mae_az = float(np.mean(abs_err_val[:, 0]))
        val_mae_el = float(np.mean(abs_err_val[:, 1]))

        self.history["val_mae_az"].append(val_mae_az)
        self.history["val_mae_el"].append(val_mae_el)

        # Training per-target MAE (optional)
        if self.x_train is not None and self.y_train is not None:
            y_pred_tr = self.model.predict(self.x_train, batch_size=32, verbose=0)
            abs_err_tr = np.abs(y_pred_tr - self.y_train)
            mae_az = float(np.mean(abs_err_tr[:, 0]))
            mae_el = float(np.mean(abs_err_tr[:, 1]))
            self.history["mae_az"].append(mae_az)
            self.history["mae_el"].append(mae_el)
        else:
            # Keep lengths consistent if train not provided
            self.history["mae_az"].append(float("nan"))
            self.history["mae_el"].append(float("nan"))

        if logs is not None:
            logs["val_mae_az"] = val_mae_az
            logs["val_mae_el"] = val_mae_el
            logs["mae_az"] = self.history["mae_az"][-1]
            logs["mae_el"] = self.history["mae_el"][-1]


@keras.saving.register_keras_serializable(package="Poirazi")
class TargetMAE(tf.keras.metrics.Metric):
    """Mean absolute error for one output target."""

    def __init__(self, target_index, name, **kwargs):
        super().__init__(name=name, **kwargs)
        self.target_index = int(target_index)
        self.total = self.add_weight(name="total", initializer="zeros")
        self.count = self.add_weight(name="count", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        error = tf.abs(y_true[:, self.target_index] - y_pred[:, self.target_index])
        if sample_weight is not None:
            sample_weight = tf.cast(sample_weight, self.dtype)
            error = tf.multiply(error, sample_weight)
            num_values = tf.reduce_sum(sample_weight)
        else:
            num_values = tf.cast(tf.size(error), self.dtype)

        self.total.assign_add(tf.reduce_sum(tf.cast(error, self.dtype)))
        self.count.assign_add(num_values)

    def result(self):
        return tf.math.divide_no_nan(self.total, self.count)

    def reset_state(self):
        self.total.assign(0.0)
        self.count.assign(0.0)

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"target_index": self.target_index})
        return cfg


def _plot_weights(model = None, run_name: str = None, run_dir: str = None, show: bool = True):
    """Visualize the learned weights of the model's spine, dendrite, and soma layers."""
    
    layers_to_plot = [model.spine_layer, model.dendrite_layer, model.soma_layer]

    fig, axes = plt.subplots(1, len(layers_to_plot), figsize=(5 * len(layers_to_plot), 5))

    for ax, layer in zip(axes, layers_to_plot):

        weights = layer.w.numpy() 
        im = ax.imshow(weights, aspect="auto", cmap="viridis")
        ax.set_title(f"{layer.name} weights")
        ax.set_xlabel("Output units")
        ax.set_ylabel("Input units")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    if run_name:
        fig.suptitle(run_name)
    plt.tight_layout()

    if run_dir is not None:
        plot_path = os.path.join(run_dir, "final_weights.png")
        fig.savefig(plot_path, dpi=150, bbox_inches="tight")
        print(f"  Saved training plot -> {plot_path}")


    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig


def plot_training_history(history: dict, run_name: str = None, run_dir: str = None, show: bool = True, accuracy: bool = False):
    """Plot the same loss and validation per-target MAE curves shown in the notebook training cell."""
    loss_values = history.get("loss", [])
    if not loss_values:
        return None

    epochs = np.arange(1, len(loss_values) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(epochs, loss_values, label="Train Loss", color="tab:blue")
    if accuracy:
        axes[0].plot(epochs, history.get("accuracy", []), label="Train Accuracy", color="tab:blue", linestyle="--")
    if "val_loss" in history:
        axes[0].plot(epochs, history["val_loss"], label="Val Loss", color="tab:orange")
    if accuracy and "val_accuracy" in history:
        axes[0].plot(epochs, history["val_accuracy"], label="Val Accuracy", color="tab:orange", linestyle="--")
    axes[0].set_title("Loss/Accuracy vs Epoch")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("MSE Loss/Accuracy")
    axes[0].set_xscale("log")
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    if "val_mae_az" in history:
        axes[1].plot(epochs, history["val_mae_az"], color="tab:orange", linestyle="--", label="Val MAE Azimuth")
    if "val_mae_el" in history:
        axes[1].plot(epochs, history["val_mae_el"], color="tab:orange", label="Val MAE Elevation")

    # Also plot training per-target MAE (dashed lines) when available
    if "mae_az" in history:
        axes[1].plot(epochs, history["mae_az"], color="tab:blue", linestyle="--", label="Train MAE Azimuth")
    if "mae_el" in history:
        axes[1].plot(epochs, history["mae_el"], color="tab:blue", label="Train MAE Elevation")

    axes[1].set_title("Validation Per-Target MAE")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("MAE (deg)")
    axes[1].set_xscale("log")
    axes[1].grid(alpha=0.3)
    axes[1].legend()

    if run_name:
        fig.suptitle(run_name)
    plt.tight_layout()

    if run_dir is not None:
        plot_path = os.path.join(run_dir, "training_history.png")
        fig.savefig(plot_path, dpi=150, bbox_inches="tight")
        print(f"  Saved training plot -> {plot_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig

# ─────────────────────────────────────────────────────────────────────────────
# Sweep configuration
# ─────────────────────────────────────────────────────────────────────────────

SWEEP_CONFIG = dict(
    spine_sizes            = [128, 256, 512],
    dendrite_per_soma_sizes = [1, 2, 4, 8, 16],
    soma_sizes             = [128, 256, 512],
)

TRAIN_CONFIG = dict(
    n_samples   = 5000,
    epochs      = 500,
    batch_size  = 32,
    lr          = 1e-3,
    test_size   = 0.2,
    random_seed = 42,
    patience = 20,
)

# Fraction of spines allocated to ITD channel; ILD gets the remainder
ITD_SPINE_FRAC_CONFIGS = {
    "equal_split":  0.5,    # 50/50 — default
    "itd_heavy":    0.67,   # ITD gets 2/3 (more complex joint tuning)
    "ild_heavy":    0.33,   # ILD gets 2/3
}

# Receptive-field overlap within each channel
OVERLAP_CONFIGS = {
    "strict": 0.0,
    "25pct":  0.25,
    "50pct":  0.50,
}

# How spines are assigned to dendrites within each channel
DENDRITE_ASSIGN_CONFIGS = [
    "topographic",   # contiguous blocks (default)
    "interleaved",   # every Nth spine -> same dendrite
    "random",        # fixed random assignment (sparsity control)
]

# Whether dendrites respect channel boundaries
CHANNEL_DEND_SPLIT_CONFIGS = {
    "split":   True,    # Ch1 spines -> Ch1 dendrites only (default)
    "merged":  False,   # all spines can project to any dendrite
}

OUTPUT_ROOT = "/home/mrsarti/rANN/sweep_results"


# ─────────────────────────────────────────────────────────────────────────────
# Masked Dense layer  (same as discussed — gradient masking via train_step)
# ─────────────────────────────────────────────────────────────────────────────

@keras.saving.register_keras_serializable(package="Poirazi")
class MaskedDense(tf.keras.layers.Layer):
    """Dense layer with a fixed boolean connectivity mask."""

    def __init__(self, units, mask=None, activation=None, l2_regularizer=0.0, weight_shape=None, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.activation = tf.keras.activations.get(activation)
        self.l2_regularizer = l2_regularizer
        self.weight_shape = weight_shape
        # Store mask as a plain numpy array so it survives serialisation
        self._mask_array = np.asarray(mask, dtype=np.float32) if mask is not None else None

    def build(self, input_shape):
        regularizer = tf.keras.regularizers.L2(self.l2_regularizer) if self.l2_regularizer > 0 else None
        kernel_initializer = "zeros" if self._mask_array is not None else "glorot_uniform"
        bias_initializer = tf.keras.initializers.Constant(0.01) if self._mask_array is not None else "zeros"
        self.w = self.add_weight(
            shape=(int(input_shape[-1]), self.units),
            initializer=kernel_initializer,
            trainable=True,
            name="kernel",
            regularizer=regularizer,
        )
        self.b = self.add_weight(
            shape=(self.units,), initializer=bias_initializer, trainable=True, name="bias"
        )
        if self._mask_array is not None:
            self.mask = self.add_weight(
                shape=self._mask_array.shape,
                initializer=tf.constant_initializer(self._mask_array),
                trainable=False,
                name="mask",
            )
            # Zero forbidden connections at initialisation
            self.w.assign(self.w * self.mask)
        else:
            # Creates a mask with all weights allowed (all-to-all)
            self.mask = self.add_weight(
                shape=self.w.shape,
                initializer=tf.constant_initializer(np.ones(self.w.shape, dtype=np.float32)),
                trainable=False,
                name="alltoall",
            )

    def call(self, inputs):
        w = self.w * self.mask #if self.mask is not None else self.w
        return self.activation(inputs @ w + self.b)

    def get_config(self):
        cfg = super().get_config()
        cfg.update(dict(
            units=self.units,
            activation=tf.keras.activations.serialize(self.activation),
            l2_regularizer=self.l2_regularizer,
            mask=self._mask_array.tolist() if self._mask_array is not None else None,
            mode = "constrained" if self.mask.name != "alltoall" else "alltoall",
        ))
        return cfg


# ─────────────────────────────────────────────────────────────────────────────
# Model class with gradient masking train_step
# ─────────────────────────────────────────────────────────────────────────────

@keras.saving.register_keras_serializable(package="Poirazi")
class rANN(tf.keras.Model):
    """
    Restricted-weight ANN for binaural sound localisation.

    Passing mask arrays activates the constrained (biologically informed)
    wiring.  Passing mask=None for all three stages gives the all-to-all
    baseline with identical layer sizes.
    """

    def __init__(
        self,
        input_dim,
        output_dim,
        n_spines    = 128, # Total number of spines (across all channels)
        n_dendrites_per_soma = None,
        n_dendrites = None,
        n_soma      = 128, # Total number of somas (output layer size)
        spine_mask    = None,
        dendrite_mask = None,
        soma_mask     = None,
        l2_reg        = 1e-4,  # L2 regularization strength
        weight_shape  = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        # Backward-compatible dendrite handling:
        # - New path: pass n_dendrites_per_soma
        # - Old saved models may only include n_dendrites (total or legacy per-soma)
        if n_dendrites_per_soma is None:
            if n_dendrites is None:
                n_dendrites_per_soma = 4
            else:
                n_dendrites_per_soma = (
                    int(n_dendrites // n_soma)
                    if n_soma and int(n_dendrites) % int(n_soma) == 0
                    else int(n_dendrites)
                )

        self.input_dim    = input_dim
        self.output_dim   = output_dim
        self.n_spines     = n_spines
        self.n_dendrites_per_soma = int(n_dendrites_per_soma)
        self.total_dendrites = self.n_dendrites_per_soma * n_soma
        self.n_dendrites = self.total_dendrites  # compatibility alias
        self.n_soma       = n_soma

        # Keep mask arrays for get_config serialisation
        self._spine_mask_arr    = spine_mask
        self._dendrite_mask_arr = dendrite_mask
        self._soma_mask_arr     = soma_mask
        self.constrained = any(mask is not None for mask in (spine_mask, dendrite_mask, soma_mask))

        # Stage 1: Toric Spines (sigmoid for supralinear toric spine integration)
        self.spine_layer = MaskedDense(
            n_spines, mask=spine_mask, activation="relu", l2_regularizer=l2_reg, name="spine_layer", weight_shape= None if weight_shape is not None else None
        )
        # Stage 2: Dendrites  
        self.dendrite_layer = MaskedDense(
            self.total_dendrites, mask=dendrite_mask, activation="relu", l2_regularizer=l2_reg, name="dendrite_layer"
        )
        # Stage 3: Soma
        self.soma_layer = MaskedDense(
            n_soma, mask=soma_mask, activation="relu", l2_regularizer=l2_reg, name="soma_layer", 
        )
        # Output: all-to-all (population code read out)
        self.out_layer = tf.keras.layers.Dense(output_dim, activation='linear', name="output_layer") # Population Code?

        # Pre-build the mask tensors used in train_step
        self._tf_spine_mask    = tf.constant(spine_mask,    dtype=tf.float32) if spine_mask    is not None else None
        self._tf_dendrite_mask = tf.constant(dendrite_mask, dtype=tf.float32) if dendrite_mask is not None else None
        self._tf_soma_mask     = tf.constant(soma_mask,     dtype=tf.float32) if soma_mask     is not None else None

        # # Test
        # self.history = {
        #     "loss": [], "accuracy": [],
        #     "val_mae_az": [], "val_mae_el": [],
        #     "mae_az": [], "mae_el": [],
        #     "val_acc_az": [], "val_acc_el": [],
        #     "acc_az": [], "acc_el": []
        # }

    def call(self, inputs, training=False):
        x = self.spine_layer(inputs)
        x = self.dendrite_layer(x)
        x = self.soma_layer(x)
        return self.out_layer(x)

    def train_step(self, data):
        x, y = data
        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)
            loss   = self.compute_loss(x=x, y=y, y_pred=y_pred, training=True)

        grads = tape.gradient(loss, self.trainable_variables)

        # Only zero gradients for the constrained wiring path.
        if self.constrained:
            masked_grads = list(grads)
            grad_by_var_id = {id(var): idx for idx, var in enumerate(self.trainable_variables)}
            masked_kernel_specs = (
                (self.spine_layer.w, self._tf_spine_mask),
                (self.dendrite_layer.w, self._tf_dendrite_mask),
                (self.soma_layer.w, self._tf_soma_mask),
            )
            for kernel_var, mask in masked_kernel_specs:
                if kernel_var is None or mask is None:
                    continue
                grad_idx = grad_by_var_id.get(id(kernel_var))
                if grad_idx is None or masked_grads[grad_idx] is None:
                    continue
                masked_grads[grad_idx] = masked_grads[grad_idx] * mask
        else:
            masked_grads = grads

        self.optimizer.apply_gradients(zip(masked_grads, self.trainable_variables))
        for metric in self.metrics:
            metric.update_state(y, y_pred)
        logs = {m.name: m.result() for m in self.metrics}
        logs["loss"] = loss
        return logs

    def test_step(self, data):
        x, y = data
        y_pred = self(x, training=False)
        loss = self.compute_loss(x=x, y=y, y_pred=y_pred, training=False)

        for metric in self.metrics:
            metric.update_state(y, y_pred)

        logs = {m.name: m.result() for m in self.metrics}
        logs["loss"] = loss
        return logs


    def inspect_train_step(self, x_batch, y_batch):
        """Run the same forward/backward pass as train_step without applying updates.

        Returns a dictionary containing raw gradients, masked gradients, and
        summary statistics for each masked kernel so the effect of the mask can
        be visualized or sanity-checked on a single batch.
        """
        x_batch = tf.convert_to_tensor(x_batch)
        y_batch = tf.convert_to_tensor(y_batch)

        with tf.GradientTape() as tape:
            y_pred = self(x_batch, training=True)
            loss = self.compute_loss(x=x_batch, y=y_batch, y_pred=y_pred, training=True)

        raw_grads = tape.gradient(loss, self.trainable_variables)
        masked_grads = list(raw_grads)
        layer_stats = []

        grad_by_var_id = {id(var): idx for idx, var in enumerate(self.trainable_variables)}
        masked_kernel_specs = (
            ("spine_layer", self.spine_layer.w, self._tf_spine_mask),
            ("dendrite_layer", self.dendrite_layer.w, self._tf_dendrite_mask),
            ("soma_layer", self.soma_layer.w, self._tf_soma_mask),
        )

        for layer_name, kernel_var, mask in masked_kernel_specs:
            if kernel_var is None or mask is None:
                continue
            grad_idx = grad_by_var_id.get(id(kernel_var))
            if grad_idx is None:
                continue

            grad = raw_grads[grad_idx]
            if grad is None:
                masked_grads[grad_idx] = None
                continue

            masked_grad = grad * mask
            masked_grads[grad_idx] = masked_grad
            forbidden = tf.equal(mask, 0.0)
            forbidden_values = tf.boolean_mask(masked_grad, forbidden)
            allowed_values = tf.boolean_mask(masked_grad, tf.logical_not(forbidden))
            layer_stats.append(dict(
                layer_name=layer_name,
                variable_name=kernel_var.name,
                raw_grad=grad.numpy(),
                masked_grad=masked_grad.numpy(),
                mask=mask.numpy(),
                raw_norm=float(tf.linalg.norm(grad).numpy()),
                masked_norm=float(tf.linalg.norm(masked_grad).numpy()),
                max_abs_forbidden_grad=float(tf.reduce_max(tf.abs(forbidden_values)).numpy()) if tf.size(forbidden_values) > 0 else 0.0,
                max_abs_allowed_grad=float(tf.reduce_max(tf.abs(allowed_values)).numpy()) if tf.size(allowed_values) > 0 else 0.0,
                forbidden_fraction=float(tf.reduce_mean(tf.cast(forbidden, tf.float32)).numpy()),
            ))

        return dict(
            loss=float(loss.numpy()),
            y_pred=y_pred.numpy(),
            raw_grads=raw_grads,
            masked_grads=masked_grads,
            layer_stats=layer_stats,
        )

    def get_config(self):
        cfg = super().get_config()
        cfg.update(dict(
            input_dim    = self.input_dim,
            output_dim   = self.output_dim,
            n_spines     = self.n_spines,
            n_dendrites_per_soma = self.n_dendrites_per_soma,
            n_dendrites  = self.total_dendrites,
            n_soma       = self.n_soma,
            l2_reg       = self.spine_layer.l2_regularizer,
            spine_mask    = self._spine_mask_arr.tolist()    if self._spine_mask_arr    is not None else None,
            dendrite_mask = self._dendrite_mask_arr.tolist() if self._dendrite_mask_arr is not None else None,
            soma_mask     = self._soma_mask_arr.tolist()     if self._soma_mask_arr     is not None else None,
        ))
        return cfg


def plot_train_step_diagnostics(diagnostics: dict, run_name: str = None, run_dir: str = None, show: bool = True):
    """Visualize the gradients that train_step computes before the optimizer update.

    This expects the output of rANN.inspect_train_step(x_batch, y_batch) and
    highlights whether masked connections are receiving zeroed gradients.
    """
    layer_stats = diagnostics.get("layer_stats", [])
    if not layer_stats:
        fig, ax = plt.subplots(figsize=(8, 2))
        ax.axis("off")
        ax.text(0.5, 0.5, "No masked kernel gradients to visualize.", ha="center", va="center")
        if run_name:
            fig.suptitle(run_name)
        if run_dir is not None:
            plot_path = os.path.join(run_dir, "train_step_diagnostics.png")
            fig.savefig(plot_path, dpi=150, bbox_inches="tight")
            print(f"  Saved train-step diagnostic plot -> {plot_path}")
        if show:
            plt.show()
        else:
            plt.close(fig)
        return fig

    n_layers = len(layer_stats)
    fig, axes = plt.subplots(n_layers, 3, figsize=(18, 4 * n_layers), squeeze=False)

    for row, stats in enumerate(layer_stats):
        layer_title = stats["layer_name"]
        raw = stats["raw_grad"]
        masked = stats["masked_grad"]
        mask = stats["mask"]
        max_abs = max(np.max(np.abs(raw)), np.max(np.abs(masked)), 1e-12)

        plots = [
            (raw, "Raw gradient", "coolwarm", -max_abs, max_abs),
            (masked, "Masked gradient", "coolwarm", -max_abs, max_abs),
            (mask, "Mask", "gray", 0.0, 1.0),
        ]

        for col, (array, title, cmap, vmin, vmax) in enumerate(plots):
            ax = axes[row, col]
            im = ax.imshow(array, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
            ax.set_title(title)
            ax.set_xlabel("Output units")
            ax.set_ylabel("Input units")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        axes[row, 0].set_ylabel(
            f"{layer_title}\nInput units",
            rotation=90,
            labelpad=18,
        )
        axes[row, 2].text(          
            1.25,
            0.5,
            (
                f"raw ||g||={stats['raw_norm']:.3e}\n"
                f"masked ||g||={stats['masked_norm']:.3e}\n"
                f"max |g| on forbidden={stats['max_abs_forbidden_grad']:.3e}\n"
                f"forbidden fraction={100.0 * stats['forbidden_fraction']:.1f}%"
            ),
            transform=axes[row, 2].transAxes,
            va="center",
            fontsize=9,
            family="monospace",
        )

    if run_name:
        fig.suptitle(f"Train-step gradient diagnostics: {run_name}")
    plt.tight_layout()

    if run_dir is not None:
        plot_path = os.path.join(run_dir, "train_step_diagnostics.png")
        fig.savefig(plot_path, dpi=150, bbox_inches="tight")
        print(f"  Saved train-step diagnostic plot -> {plot_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Data generation  (mirrors your notebook exactly)
# ─────────────────────────────────────────────────────────────────────────────

def _make_dataset(n_samples, seed=42):
    """Import and call create_synthetic_dataset from your notebook environment.
    Falls back to a minimal inline version if the notebook function is not
    importable, so this file is self-contained for testing."""
    try:
        from rANN.make_data import create_synthetic_dataset 
        x, y, *_ = create_synthetic_dataset(n_samples=n_samples, itd_hwhh=0.03)
    except ImportError:
        # ── Minimal inline fallback (matches notebook defaults) ───────────────
        from sklearn.preprocessing import StandardScaler
        rng = np.random.default_rng(seed)
        N_TIME, N_FREQ, N_ITD, N_ILD = 8, 32, 32, 32
        n_feat = N_TIME * (N_FREQ + N_ITD + N_ILD)
        x = rng.standard_normal((n_samples, n_feat)).astype(np.float32)
        y = rng.uniform([-90, -45], [90, 45], size=(n_samples, 2)).astype(np.float32)

    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)
    x_tr, x_te, y_tr, y_te = train_test_split(
        x_scaled, y, test_size=TRAIN_CONFIG["test_size"],
        random_state=TRAIN_CONFIG["random_seed"]
    )
    return x_tr, x_te, y_tr, y_te


# ─────────────────────────────────────────────────────────────────────────────
# Single run
# ─────────────────────────────────────────────────────────────────────────────

def _run_one(
    run_name: str,
    run_dir:  str,
    x_tr, x_te, y_tr, y_te,
    n_spines:               int,
    n_dendrites_per_soma:   int,
    n_soma:                 int,
    spine_mask    = None,
    dendrite_mask = None,
    soma_mask     = None,
    epochs:     int = 500,
    batch_size: int = 32,
    lr:         float = 1e-3,
    l2_reg:     float = 1e-4,
    mask_summary: dict = None,
    topology_png:  str = None,
    arch_png:      str = None,
    plot_history:  bool = False,
    plot_weights: bool = False,
    patience:     int = 20,
) -> dict:
    """Train one model, save it, return a metrics dict for the summary row."""

    os.makedirs(run_dir, exist_ok=True)
    input_dim  = x_tr.shape[1]
    output_dim = y_tr.shape[1]
    constrained = spine_mask is not None
    total_dendrites = n_dendrites_per_soma * n_soma

    print(x_tr.shape, y_tr.shape, x_te.shape, y_te.shape)

    # Convert to TensorFlow datasets
    train_dataset = tf.data.Dataset.from_tensor_slices((x_tr, y_tr)).batch(batch_size)
    test_dataset = tf.data.Dataset.from_tensor_slices((x_te, y_te)).batch(batch_size)

    # ── Build model ───────────────────────────────────────────────────────────
    model = rANN(
        input_dim    = input_dim,
        output_dim   = output_dim,
        n_spines     = n_spines,
        n_dendrites_per_soma = n_dendrites_per_soma,
        n_soma       = n_soma,
        spine_mask    = spine_mask,
        dendrite_mask = dendrite_mask,
        soma_mask     = soma_mask,
        l2_reg        = l2_reg,
    )
    model.compile(
        optimizer = 'adam', #tf.keras.optimizers.Adam(learning_rate=lr),
        loss      = tf.keras.losses.MeanSquaredError(),
        metrics   = [
            tf.keras.metrics.MeanAbsoluteError(name="mae"),
            tf.keras.metrics.Accuracy(name="accuracy"),
            TargetMAE(0, name="mae_az"),
            TargetMAE(1, name="mae_el"),
        ],
    )
    # # Force build
    # model(x_tr[:1])

    # ── Train ─────────────────────────────────────────────────────────────────
    t_start = time.perf_counter()

    # Early stopping: stop if val_loss plateaus (prevent overfitting) and restore best weights at the end
    if patience is not None and patience > 0:
        early_stop = tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=patience,           # stop if no improvement for 20 epochs
            restore_best_weights=True,
            verbose=0,
        )   
    
    ptcb = PerTargetMAECallback(x_te, y_te, x_train=x_tr, y_train=y_tr)

    hist = model.fit(
        train_dataset, 
        validation_data=test_dataset, 
        epochs=epochs, 
        batch_size=batch_size,
        verbose=0,
        callbacks=[early_stop] if patience is not None and patience > 0 else [ptcb],
        )

    wall_time = time.perf_counter() - t_start

    # ── Per-target MAE ────────────────────────────────────────────────────────
    y_pred   = model.predict(x_te, verbose=0)
    abs_err  = np.abs(y_pred - y_te)
    mae_az   = float(np.mean(abs_err[:, 0]))
    mae_el   = float(np.mean(abs_err[:, 1]))

    print(f"  {run_name:<40}  val_loss={hist.history['val_loss'][-1]:.3f}  mae_az={mae_az:.2f}°  mae_el={mae_el:.2f}°  time={wall_time:.1f}s")

    # Epoch at which val_loss first reached 90% of its final improvement
    val_losses = hist.history["val_loss"]
    best_val   = min(val_losses)
    threshold  = val_losses[0] - 0.9 * (val_losses[0] - best_val)
    epoch_90   = next((i + 1 for i, v in enumerate(val_losses) if v <= threshold), epochs)

    # ── Save model ────────────────────────────────────────────────────────────
    model.save(os.path.join(run_dir, "model.keras"))

    # ── Save history ──────────────────────────────────────────────────────────
    # If per-target validation MAE was recorded, also save an aggregated
    # `val_mae` (mean of az/el) so older tools that expect `val_mae`
    # continue to work.
    if "val_mae_az" in hist.history and "val_mae_el" in hist.history:
        az = hist.history["val_mae_az"]
        el = hist.history["val_mae_el"]
        hist.history["val_mae"] = [(a + e) / 2.0 for a, e in zip(az, el)]

    if plot_history:
        plot_training_history(hist.history, run_name=run_name, run_dir=run_dir, show=True)

    if plot_weights:
        _plot_weights(model, run_name=run_name, run_dir=run_dir, show=True)

    with open(os.path.join(run_dir, "history.json"), "w") as f:
        json.dump({k: [float(v) for v in vs] for k, vs in hist.history.items()}, f, indent=2)

    # ── Save config ───────────────────────────────────────────────────────────
    cfg = dict(
        run_name    = run_name,
        constrained = constrained,
        n_spines    = n_spines,
        n_dendrites_per_soma = n_dendrites_per_soma,
        total_dendrites      = total_dendrites,
        n_soma      = n_soma,
        input_dim   = input_dim,
        output_dim  = output_dim,
        epochs      = epochs,
        batch_size  = batch_size,
        lr          = lr,
        trainable_params = int(np.sum([np.prod(v.shape) for v in model.trainable_variables])),
        mask_summary = mask_summary or {},
        topology_png = topology_png,
        arch_png     = arch_png,
    )
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2)

    print(
        f"  {run_name:<40}  "
        f"val_loss={best_val:.3f}  "
        f"mae_az={mae_az:.2f}°  mae_el={mae_el:.2f}°  "
        f"epoch_90={epoch_90:>4}  "
        f"time={wall_time:.1f}s"
    )

    return dict(
        run_name        = run_name,
        run_dir         = run_dir,
        constrained     = constrained,
        n_spines        = n_spines,
        n_dendrites_per_soma = n_dendrites_per_soma,
        total_dendrites = total_dendrites,
        n_soma          = n_soma,
        trainable_params= cfg["trainable_params"],
        active_weights  = mask_summary["active_weights"]["total"]   if mask_summary else None,
        all2all_weights = mask_summary["all2all_weights"]["total"]   if mask_summary else None,
        weight_reduction= mask_summary["weight_reduction_pct"]       if mask_summary else None,
        density_overall = mask_summary["density"]["overall"]         if mask_summary else None,
        topology_png    = topology_png,
        arch_png        = arch_png,
        final_train_loss= float(hist.history["loss"][-1]),
        final_val_loss  = best_val,
        final_mae_az    = mae_az,
        final_mae_el    = mae_el,
        epoch_90pct     = epoch_90,
        wall_time_s     = round(wall_time, 2),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Helpers for sweep loop (reduce duplication)
# ─────────────────────────────────────────────────────────────────────────────

def _generate_run_figures(
    mode, n_spines, n_dendrites_per_soma, n_soma, run_dir, output_root,
    n_time=None, n_freq=None, n_itd=None, n_ild=None,
    itd_spine_frac=None, overlap=None, dendrite_rule=None, channel_dend_split=None,
    plot_mask=None, plot_architecture=None,
    file_suffix="",
):
    """Generate topology and architecture plots, return relative paths."""
    topology_png_rel = arch_png_rel = None
    
    #print(plot_mask, plot_architecture)

    if plot_mask is not None:
        png_path = plot_masks(
            n_spines, n_dendrites_per_soma, n_soma,
            outdir=run_dir, save=True, mode=mode,
            n_time=n_time, n_freq=n_freq, n_itd=n_itd, n_ild=n_ild,
            itd_spine_frac=itd_spine_frac, overlap=overlap,
            dendrite_rule=dendrite_rule, channel_dend_split=channel_dend_split,
            file_suffix=file_suffix,
        )
        topology_png_rel = os.path.relpath(png_path, output_root)
    if plot_architecture is not None:
        arch_path = plot_architectures(
            n_spines, n_dendrites_per_soma, n_soma,
            outdir=run_dir, save=True, mode=mode,
            itd_spine_frac=itd_spine_frac, overlap=overlap,
            dendrite_rule=dendrite_rule, channel_dend_split=channel_dend_split,
            file_suffix=file_suffix,
        )
        arch_png_rel = os.path.relpath(arch_path, output_root)

    return topology_png_rel, arch_png_rel


def _load_or_train_model(
    tag, run_dir, x_tr, x_te, y_tr, y_te,
    n_spines, n_dendrites_per_soma, n_soma, total_dendrites,
    spine_mask, dendrite_mask, soma_mask,
    epochs, batch_size, lr, l2_reg, patience,
    mask_summary, topology_png_rel, arch_png_rel, plot_history, plot_weights
):
    """Train or load a model, return full row dict for the summary."""
    if os.path.exists(os.path.join(run_dir, "model.keras")):
        # Load saved model and metrics
        print(f"  Skipping (already exists) — loading metrics from config.json")
        with open(os.path.join(run_dir, "config.json")) as f:
            saved_cfg = json.load(f)
        with open(os.path.join(run_dir, "history.json")) as f:
            hist_data = json.load(f)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            y_pred = tf.keras.models.load_model(run_dir + "/model.keras")(x_te).numpy()
        abs_err = np.abs(y_pred - y_te)
        val_losses = hist_data["val_loss"]
        best_val = min(val_losses)
        threshold = val_losses[0] - 0.9 * (val_losses[0] - best_val)
        epoch_90 = next((i + 1 for i, v in enumerate(val_losses) if v <= threshold), len(val_losses))
        if plot_history:
            plot_training_history(hist_data, run_name=tag, run_dir=run_dir, show=False)
        return dict(
            run_name=tag,
            run_dir=run_dir,
            trainable_params=saved_cfg.get("trainable_params"),
            active_weights=mask_summary["active_weights"]["total"] if mask_summary else None,
            all2all_weights=mask_summary["all2all_weights"]["total"] if mask_summary else None,
            weight_reduction=mask_summary["weight_reduction_pct"] if mask_summary else None,
            density_overall=mask_summary["density"]["overall"] if mask_summary else None,
            topology_png=topology_png_rel,
            arch_png=arch_png_rel,
            final_train_loss=hist_data["loss"][-1],
            final_val_loss=best_val,
            final_mae_az=float(np.mean(abs_err[:, 0])),
            final_mae_el=float(np.mean(abs_err[:, 1])),
            epoch_90pct=epoch_90,
            wall_time_s=None,
        )
    else:
        # Train new model
        row = _run_one(
            run_name=tag, run_dir=run_dir,
            x_tr=x_tr, x_te=x_te, y_tr=y_tr, y_te=y_te,
            n_spines=n_spines, n_dendrites_per_soma=n_dendrites_per_soma, n_soma=n_soma,
            spine_mask=spine_mask, dendrite_mask=dendrite_mask, soma_mask=soma_mask,
            epochs=epochs, batch_size=batch_size, lr=lr, l2_reg=l2_reg, patience=patience,
            mask_summary=mask_summary, topology_png=topology_png_rel, arch_png=arch_png_rel,
            plot_history=plot_history, plot_weights=plot_weights,
        )
        return row


# ─────────────────────────────────────────────────────────────────────────────
# Main sweep loop
# ─────────────────────────────────────────────────────────────────────────────

def run_sweep(
    spine_sizes:             list  = None,
    dendrite_per_soma_sizes: list  = None,
    soma_sizes:              list  = None,
    epochs:                  int   = None,
    n_samples:               int   = None,
    batch_size:              int   = None,
    lr:                      float = None,
    l2_reg:                  float = 1e-4,
    patience:               int   = None,
    output_root:             str   = OUTPUT_ROOT,
    include_baseline:        bool  = True,    # also train all-to-all for each config

    # Defaults for dataset dimensions (passed to sweep_masks for mask generation)
    n_time=8, n_freq=32, n_itd=32, n_ild=32,

    # Defaults for other mask config dimensions
    itd_spine_frac_names=None, # ['equal_split', 'itd_heavy']
    overlap_names=None, # ['strict', '25pct']
    dendrite_rules=None, # ['topographic', 'interleaved']
    channel_dend_split_names=None, # ['split', 'merged']
    plot_history:        bool = False,
    plot_weights:        bool = False,
    plot_mask:          callable = True, 
    plot_architecture:   callable = True,  # function to generate and save architecture visualizations for each config (called during each run with config params and run_dir)
) -> pd.DataFrame:
    """
    Run the full sweep and return a summary DataFrame.

    Each (n_spines, n_dendrites_per_soma, n_soma) config trains two models:
        • constrained  — biological boolean masks applied
        • alltoall     — same layer sizes, no masks (baseline)

    Parameters
    ----------
    dendrite_per_soma_sizes : list of dendrite counts per soma
    include_baseline : if False, skip the all-to-all runs (faster, no comparison)
    """

    # Apply defaults
    spine_sizes              = spine_sizes              or SWEEP_CONFIG["spine_sizes"]
    dendrite_per_soma_sizes  = dendrite_per_soma_sizes  or SWEEP_CONFIG["dendrite_per_soma_sizes"]
    soma_sizes               = soma_sizes               or SWEEP_CONFIG["soma_sizes"]
    epochs         = epochs         or TRAIN_CONFIG["epochs"]
    n_samples      = n_samples      or TRAIN_CONFIG["n_samples"]
    batch_size     = batch_size     or TRAIN_CONFIG["batch_size"]
    lr             = lr             or TRAIN_CONFIG["lr"]
    # Respect explicit None (disable early stopping) instead of forcing defaults.

    runs_dir     = os.path.join(output_root, "runs")
    os.makedirs(runs_dir, exist_ok=True)

    # ── Generate data once (shared across all runs) ───────────────────────────
    print(f"Generating {n_samples} synthetic samples...")
    x_tr, x_te, y_tr, y_te = _make_dataset(n_samples)
    print(f"  train: {x_tr.shape}  test: {x_te.shape}\n")

    # ── Build all masks upfront ───────────────────────────────────────────────
    print("Building biological masks...")
    all_masks = sweep_masks(
        spine_sizes              = spine_sizes,
        dendrite_per_soma_sizes = dendrite_per_soma_sizes,
        soma_sizes              = soma_sizes,
        n_time=n_time, 
        n_freq=n_freq, 
        n_itd=n_itd, 
        n_ild=n_ild,
        # Don't generate figures for every config here (slow) — we'll generate them on demand during each run and save them into the run folder

        # Defaults for other mask config dimensions
        itd_spine_frac_names=itd_spine_frac_names or ['equal_split', 'itd_heavy'],
        overlap_names=overlap_names or ['strict', '25pct'],
        dendrite_rules=dendrite_rules or ['topographic', 'interleaved'],
        channel_dend_split_names=channel_dend_split_names or ['split', 'merged']
    )
    n_configs = len(all_masks)
    # Count unique size tuples so baseline (all-to-all) runs occur once per size
    unique_sizes = { (key[-3], key[-2], key[-1]) for key in all_masks.keys() }
    n_baselines = len(unique_sizes) if include_baseline else 0
    n_runs = n_configs + n_baselines
    print(f"  {n_configs} topology configs  ·  {n_baselines} unique sizes (baselines) = {n_runs} total runs\n")

    rows = []
    run_idx = 0

    # Track which size tuples have had their all-to-all baseline run
    seen_baseline_sizes = set()

    for key, masks in sorted(all_masks.items(), key=lambda item: item[0][-3:]):
        *extra, n_spines, n_dendrites_per_soma, n_soma = key

        # Optional topology metadata from sweep_masks key
        if len(extra) >= 4:
            itd_frac_name, overlap_name, dendrite_rule, channel_dend_split_name = extra[:4]
        else:
            itd_frac_name = "equal_split"
            overlap_name = "strict"
            dendrite_rule = "topographic"
            channel_dend_split_name = "split"

        topo_meta = dict(
            itd_frac_name=itd_frac_name,
            overlap_name=overlap_name,
            dendrite_rule=dendrite_rule,
            channel_dend_split_name=channel_dend_split_name,
        )
        topo_tag = (
            f"_if{itd_frac_name}_ov{overlap_name}"
            f"_dr{dendrite_rule}_cd{channel_dend_split_name}"
        )

        print(
            f"\nConfig: spines={n_spines}, dendrites/soma={n_dendrites_per_soma}, "
            f"somas={n_soma}, topology={topo_meta}"
        )
        summary = masks["summary"]

        # Geometry for this configuration
        total_dendrites = n_dendrites_per_soma * n_soma

        # --- Constrained run for this topology (always run) -----------------
        mode = "constrained"
        run_idx += 1
        tag      = f"{mode}_s{n_spines}_d{n_dendrites_per_soma}ps_so{n_soma}{topo_tag}"
        run_dir  = os.path.join(runs_dir, tag)
        os.makedirs(run_dir, exist_ok=True)

        # Generate topology-specific figures into the run folder
        topology_png_rel, arch_png_rel = _generate_run_figures(
            mode="constrained",
            n_time=n_time, n_freq=n_freq, n_itd=n_itd, n_ild=n_ild,
            n_spines=n_spines, n_dendrites_per_soma=n_dendrites_per_soma, n_soma=n_soma,
            run_dir=run_dir, output_root=output_root,
            itd_spine_frac=ITD_SPINE_FRAC_CONFIGS[itd_frac_name],
            overlap=OVERLAP_CONFIGS[overlap_name],
            dendrite_rule=dendrite_rule,
            channel_dend_split=CHANNEL_DEND_SPLIT_CONFIGS[channel_dend_split_name],
            plot_mask=plot_mask,
            plot_architecture=plot_architecture,
            file_suffix=(
                f"if{itd_frac_name}_ov{overlap_name}"
                f"_dr{dendrite_rule}_cd{channel_dend_split_name}"
            ),
        )


        # Verify figures match the mask config immediately after generation
        if verify_figure_masks is not None and topology_png_rel is not None:
            _vresult = verify_figure_masks(
                n_spines             = n_spines,
                n_dendrites_per_soma = n_dendrites_per_soma,
                n_soma               = n_soma,
                run_dir              = run_dir,
                itd_spine_frac       = ITD_SPINE_FRAC_CONFIGS[itd_frac_name],
                overlap              = OVERLAP_CONFIGS[overlap_name],
                dendrite_rule        = dendrite_rule,
                channel_dend_split   = CHANNEL_DEND_SPLIT_CONFIGS[channel_dend_split_name],
            )
            if not _vresult["ok"]:
                print(f"  [VERIFY FAILED] {tag}")
                if print_verification is not None:
                    print_verification(_vresult)
            else:
                print(f"  [VERIFY OK]  active={_vresult['fingerprint']['active_total']:,}  "
                        f"reduction={100*(1-_vresult['fingerprint']['active_total']/_vresult['fingerprint'].get('active_total',1)):.0f}%  "
                        f"bio_rules=PASS")

        print(f"[{run_idx}/{n_runs}] {tag}")

        row = _load_or_train_model(
            tag=tag, run_dir=run_dir,
            x_tr=x_tr, x_te=x_te, y_tr=y_tr, y_te=y_te,
            n_spines=n_spines, n_dendrites_per_soma=n_dendrites_per_soma, n_soma=n_soma,
            total_dendrites=total_dendrites,
            spine_mask=masks["input_spine"], dendrite_mask=masks["spine_dendrite"], soma_mask=masks["dendrite_soma"],
            epochs=epochs, batch_size=batch_size, lr=lr, l2_reg=l2_reg, patience=patience,
            mask_summary=summary, topology_png_rel=topology_png_rel, arch_png_rel=arch_png_rel,
            plot_history=plot_history, plot_weights=plot_weights,
        )
        row.update(dict(
            constrained=True,
            itd_frac_name=itd_frac_name,
            overlap_name=overlap_name,
            dendrite_rule=dendrite_rule,
            channel_dend_split_name=channel_dend_split_name,
            n_spines=n_spines,
            n_dendrites_per_soma=n_dendrites_per_soma,
            total_dendrites=total_dendrites,
            n_soma=n_soma,
        ))
        rows.append(row)

        # --- Baseline all-to-all: run once per unique size ------------------
        size_key = (n_spines, n_dendrites_per_soma, n_soma)
        if include_baseline and size_key not in seen_baseline_sizes:
            seen_baseline_sizes.add(size_key)
            mode = "alltoall"
            run_idx += 1
            # baseline tag omits topology suffix so it is unique per size
            tag = f"{mode}_s{n_spines}_d{n_dendrites_per_soma}ps_so{n_soma}"
            run_dir = os.path.join(runs_dir, tag)
            os.makedirs(run_dir, exist_ok=True)

            topology_png_rel, arch_png_rel = _generate_run_figures(
                mode="alltoall",
                n_spines=n_spines, n_dendrites_per_soma=n_dendrites_per_soma, n_soma=n_soma,
                n_time=n_time, n_freq=n_freq, n_itd=n_itd, n_ild=n_ild,
                run_dir=run_dir, output_root=output_root, plot_mask=plot_mask, plot_architecture=plot_architecture, 
            )

            print(f"[{run_idx}/{n_runs}] {tag}")

            row = _load_or_train_model(
                tag=tag, run_dir=run_dir,
                x_tr=x_tr, x_te=x_te, y_tr=y_tr, y_te=y_te,
                n_spines=n_spines, n_dendrites_per_soma=n_dendrites_per_soma, n_soma=n_soma,
                total_dendrites=total_dendrites,
                spine_mask=None, dendrite_mask=None, soma_mask=None,
                epochs=epochs, batch_size=batch_size, lr=lr, l2_reg=l2_reg, patience=patience,
                mask_summary=None, topology_png_rel=topology_png_rel, arch_png_rel=arch_png_rel,
                plot_history=plot_history, plot_weights=plot_weights,
            )
            row.update(dict(
                constrained=False,
                itd_frac_name=None,
                overlap_name=None,
                dendrite_rule=None,
                channel_dend_split_name=None,
                n_spines=n_spines,
                n_dendrites_per_soma=n_dendrites_per_soma,
                total_dendrites=total_dendrites,
                n_soma=n_soma,
            ))
            rows.append(row)

            
    df = pd.DataFrame(rows)

    # Sort by constrained first, then config size
    sort_cols = [
        "itd_frac_name", "overlap_name", "dendrite_rule", "channel_dend_split_name",
        "n_spines", "n_dendrites_per_soma", "n_soma", "constrained"
    ]
    sort_cols = [c for c in sort_cols if c in df.columns]
    asc = [True] * len(sort_cols)
    if "constrained" in sort_cols:
        asc[sort_cols.index("constrained")] = False
    df = df.sort_values(sort_cols, ascending=asc).reset_index(drop=True)

    # Convenience: improvement of constrained over all-to-all for each config
    if include_baseline:
        df = _add_comparison_columns(df)

    # Save
    csv_path = os.path.join(output_root, "sweep_summary.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nSummary saved → {csv_path}")
    print(f"DataFrame shape: {df.shape}")

    return df



def _add_comparison_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each (n_spines, n_dendrites_per_soma, n_soma) pair, compute the delta between
    constrained and all-to-all runs so you can directly read off the effect.
    """
    group_cols = ["n_spines", "n_dendrites_per_soma", "n_soma"]
    for c in ["itd_frac_name", "overlap_name", "dendrite_rule", "channel_dend_split_name"]:
        if c in df.columns:
            group_cols.append(c)

    groups = df.groupby(group_cols, dropna=False)

    for _, grp in groups:
        c = grp[grp["constrained"] == True]
        a = grp[grp["constrained"] == False]
        if c.empty or a.empty:
            continue
        ci = c.index[0]
        ai = a.index[0]
        df.loc[ci, "delta_val_loss"]   = df.loc[ai, "final_val_loss"]  - df.loc[ci, "final_val_loss"]
        df.loc[ci, "delta_mae_az"]     = df.loc[ai, "final_mae_az"]    - df.loc[ci, "final_mae_az"]
        df.loc[ci, "delta_mae_el"]     = df.loc[ai, "final_mae_el"]    - df.loc[ci, "final_mae_el"]
        df.loc[ci, "delta_epoch_90"]   = df.loc[ai, "epoch_90pct"]     - df.loc[ci, "epoch_90pct"]
        df.loc[ci, "delta_wall_time_s"]= df.loc[ai, "wall_time_s"]     - df.loc[ci, "wall_time_s"]

    return df


# ─────────────────────────────────────────────────────────────────────────────
# Utility: reload summary from saved runs (no retraining)
# ─────────────────────────────────────────────────────────────────────────────

def load_summary(output_root: str = OUTPUT_ROOT) -> pd.DataFrame:
    """
    Reconstruct the summary DataFrame by scanning saved run directories.
    Useful if you want to reload results in a new session without re-running.

        df = load_summary("sweep_results")
    """
    csv_path = os.path.join(output_root, "sweep_summary.csv")
    if os.path.exists(csv_path):
        return pd.read_csv(csv_path)

    runs_dir = os.path.join(output_root, "runs")
    rows = []
    for run_name in sorted(os.listdir(runs_dir)):
        run_dir  = os.path.join(runs_dir, run_name)
        cfg_path = os.path.join(run_dir, "config.json")
        his_path = os.path.join(run_dir, "history.json")
        if not (os.path.exists(cfg_path) and os.path.exists(his_path)):
            continue
        with open(cfg_path) as f:
            cfg = json.load(f)
        with open(his_path) as f:
            hist = json.load(f)
        # Compute final_val_mae: prefer `val_mae`, else derive from per-target keys
        if "val_mae" in hist:
            final_val_mae = min(hist.get("val_mae", [float("nan")]))
        elif "val_mae_az" in hist and "val_mae_el" in hist:
            vals = [(a + e) / 2.0 for a, e in zip(hist.get("val_mae_az", []), hist.get("val_mae_el", []))]
            final_val_mae = min(vals) if vals else float("nan")
        else:
            final_val_mae = float("nan")

        rows.append(dict(
            run_name         = run_name,
            run_dir          = run_dir,
            constrained      = cfg.get("constrained"),
            n_spines         = cfg.get("n_spines"),
            n_dendrites      = cfg.get("n_dendrites"),
            n_soma           = cfg.get("n_soma"),
            trainable_params = cfg.get("trainable_params"),
            active_weights   = cfg.get("mask_summary", {}).get("active_weights", {}).get("total"),
            weight_reduction = cfg.get("mask_summary", {}).get("weight_reduction_pct"),
            final_train_loss = hist["loss"][-1],
            final_val_loss   = min(hist["val_loss"]),
            final_val_mae    = final_val_mae,
        ))

    df = pd.DataFrame(rows)
    df.to_csv(csv_path, index=False)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────


if __name__ == "__main__":
    df = run_sweep(
        spine_sizes             = [32, 64],
        dendrite_per_soma_sizes = [4],          
        soma_sizes              = [32, 64],
        epochs                  = 100,          # increase to 500 for final runs
        n_samples               = 1000,         # increase to 5000 for final runs
        batch_size              = 32,
        lr                      = 1e-3,
        include_baseline        = True,
    )
    # df = run_sweep(epochs=500, n_samples=5000)
    print("\n── Top 10 constrained runs by val_loss ──")
    cols = ["run_name", "n_spines", "n_dendrites_per_soma", "total_dendrites", "n_soma",
            "final_val_loss", "final_mae_az", "final_mae_el",
            "epoch_90pct", "weight_reduction", "wall_time_s"]
    print(df[df["constrained"] == True][cols].head(10).to_string(index=False))

