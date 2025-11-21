# train_regression.py

import time
from typing import Iterator, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.training import train_state
import tensorflow_datasets as tfds

from ckhronos import ConvKHRONOSRegressor, count_parameters


class TrainState(train_state.TrainState):
    """null."""

def load_mnist_splits() -> Tuple[np.ndarray, ...]:

    ds_train = tfds.load("mnist", split="train", as_supervised=False)
    ds_test = tfds.load("mnist", split="test", as_supervised=False)

    train_np = tfds.as_numpy(ds_train.batch(60_000))
    test_np = tfds.as_numpy(ds_test.batch(10_000))

    train_data = next(iter(train_np))
    test_data = next(iter(test_np))

    images = train_data["image"].astype(np.float32) / 255.0
    labels = train_data["label"].astype(np.float32)

    rng = np.random.default_rng(0)
    perm = rng.permutation(images.shape[0])
    images = images[perm]
    labels = labels[perm]

    images_train, images_val = images[:55_000], images[55_000:]
    labels_train, labels_val = labels[:55_000], labels[55_000:]

    images_test = test_data["image"].astype(np.float32) / 255.0
    labels_test = test_data["label"].astype(np.float32)

    return images_train, labels_train, images_val, labels_val, images_test, labels_test


def make_batches(
        images: jnp.ndarray, labels: jnp.ndarray, batch_size: int, rng: jax.random.PRNGKey
        ) -> Iterator[Tuple[jnp.ndarray, jnp.ndarray]]:

    perm = jax.random.permutation(rng, images.shape[0])
    images = images[perm]
    labels = labels[perm]
    n_batches = images.shape[0] // batch_size
    for i in range(n_batches):
        start = i * batch_size
        end = start + batch_size
        yield images[start:end], labels[start:end]


def create_train_state(
        model: ConvKHRONOSRegressor, rng: jax.random.PRNGKey, sample_shape: Tuple[int, int, int], lr_peak: float = 3e-4
        ) -> TrainState:

    dummy = jnp.ones((1, *sample_shape), model.compute_dtype)
    params = model.init(rng, dummy)["params"]
    schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=lr_peak,
        warmup_steps=300,
        decay_steps=5000,
        end_value=1e-4,
    )
    tx = optax.adam(schedule)
    return TrainState.create(apply_fn=model.apply, params=params, tx=tx)


@jax.jit
def train_step(state: TrainState, batch_images, batch_targets):

    def loss_fn(params):
        preds = state.apply_fn({"params": params}, batch_images.astype(jnp.float32))
        preds = preds.astype(jnp.float32)
        loss = jnp.mean((preds - batch_targets) ** 2)
        return loss, preds

    (loss, preds), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
    state = state.apply_gradients(grads=grads)
    mae = jnp.mean(jnp.abs(preds - batch_targets))
    return state, loss, mae


@jax.jit
def eval_step(state: TrainState, images, targets):
    preds = state.apply_fn({"params": state.params}, images.astype(jnp.float32))
    preds = preds.astype(jnp.float32)
    mse = jnp.mean((preds - targets) ** 2)
    mae = jnp.mean(jnp.abs(preds - targets))
    return mse, mae


def main():
    images_train_np, labels_train_np, images_val_np, labels_val_np, images_test_np, labels_test_np = load_mnist_splits()
    images_train = jnp.asarray(images_train_np, dtype=jnp.float32)
    labels_train = jnp.asarray(labels_train_np, dtype=jnp.float32)
    images_val = jnp.asarray(images_val_np, dtype=jnp.float32)
    labels_val = jnp.asarray(labels_val_np, dtype=jnp.float32)
    images_test = jnp.asarray(images_test_np, dtype=jnp.float32)
    labels_test = jnp.asarray(labels_test_np, dtype=jnp.float32)
    print(
        f"Train: {images_train.shape[0]}, Val: {images_val.shape[0]}, Test: {images_test.shape[0]} | "
        f"image shape {images_train.shape[1:]}"
    )

    model = ConvKHRONOSRegressor(kdims=32, kelem=8, krank=32, compute_dtype=jnp.float32)
    rng = jax.random.PRNGKey(123)
    rng, init_rng = jax.random.split(rng)
    state = create_train_state(model, init_rng, images_train.shape[1:], lr_peak=3e-4)
    print(f"Initialized regressor with {count_parameters(state.params):,} parameters.")

    num_epochs = 10
    batch_size = 128

    for epoch in range(1, num_epochs + 1):
        epoch_start = time.time()
        rng, batch_rng = jax.random.split(rng)
        total_loss = 0.0
        total_mae = 0.0
        steps = 0

        for batch_images, batch_targets in make_batches(images_train, labels_train, batch_size, batch_rng):
            state, loss, mae = train_step(state, batch_images, batch_targets)
            total_loss += float(loss)
            total_mae += float(mae)
            steps += 1

        avg_loss = total_loss / steps
        avg_mae = total_mae / steps
        val_mse, val_mae = eval_step(state, images_val, labels_val)
        epoch_time = time.time() - epoch_start
        print(
            f"Epoch {epoch:02d} | {epoch_time:4.1f}s | "
            f"train mse {avg_loss:.4f}, train mae {avg_mae:.4f} | "
            f"val mse {float(val_mse):.4f}, val mae {float(val_mae):.4f}"
        )

    test_mse, test_mae = eval_step(state, images_test, labels_test)
    print(f"Done. Test MSE {float(test_mse):.4f}, Test MAE {float(test_mae):.4f}")


if __name__ == "__main__":
    main()
