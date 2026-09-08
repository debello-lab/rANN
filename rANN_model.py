import numpy as np
import tensorflow as tf
import keras

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
        spine_activation = "relu",
        dendrite_activation = "relu",
        soma_activation = "relu",
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

        self.spine_activation = spine_activation
        self.dendrite_activation = dendrite_activation
        self.soma_activation = soma_activation
    

        # Keep mask arrays for get_config serialisation
        self._spine_mask_arr    = spine_mask
        self._dendrite_mask_arr = dendrite_mask
        self._soma_mask_arr     = soma_mask
        self.constrained = any(mask is not None for mask in (spine_mask, dendrite_mask, soma_mask))

        # Stage 1: Toric Spines (sigmoid for supralinear toric spine integration)
        self.spine_layer = MaskedDense(
            n_spines, mask=spine_mask, activation=spine_activation, l2_regularizer=l2_reg, name="spine_layer", weight_shape= None if weight_shape is not None else None
        )
        # Stage 2: Dendrites  
        self.dendrite_layer = MaskedDense(
            self.total_dendrites, mask=dendrite_mask, activation=dendrite_activation, l2_regularizer=l2_reg, name="dendrite_layer"
        )
        # Stage 3: Soma
        self.soma_layer = MaskedDense(
            n_soma, mask=soma_mask, activation=soma_activation, l2_regularizer=l2_reg, name="soma_layer", 
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