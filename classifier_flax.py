import os
import jax
import jax.numpy as jnp
import optax
from flax import linen as nn
from flax.training import train_state
import tensorflow_datasets as tfds
import numpy as np
from functools import partial
from aggregate_jax import jax_scan_aggregate
from tqdm import tqdm
import logging 

jax.clear_caches()

# Set memory growth for GPU
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class ClassificationModel(nn.Module):
    n: int   # parameter for aggregate (assumed to be part of image transformation)
    p: int   # additional rows
    q: int   # additional columns

    def setup(self):
        # Define the feed-forward network layers.
        # The input dimension is (n+p)*(n+q) after aggregation and flattening.
        self.dense1 = nn.Dense(32)
        self.dense2 = nn.Dense(10)  # for 10 MNIST classes

        self.a = self.param('a', lambda rng: jnp.ones((self.n,)))
        self.b = self.param('b', lambda rng: jnp.ones((self.n,)))
        self.b_prime = self.param('b_prime', lambda rng: jnp.ones((self.n,)))
        self.c = self.param('c', lambda rng: jnp.ones((self.n,)))
        self.c_prime = self.param('c_prime', lambda rng: jnp.ones((self.n,)))
        self.d = self.param('d', lambda rng: jnp.ones((self.n,)))
        self.d_prime = self.param('d_prime', lambda rng: jnp.ones((self.n,)))
        self.e = self.param('e', lambda rng: jnp.ones((self.q,)))
            
    def __call__(self, images):
        # Define model parameters (a, b, c, d, e) using self.param.
        params_dict = {
            'a': self.a,
            'b': self.b,
            'b_prime' : self.b_prime,
            'c': self.c,
            'c_prime' : self.c_prime,
            'd': self.d,
            'd_prime' : self.d_prime,
            'e': self.e,
        }
        # Get the aggregated representation.
        aggregate = jax_scan_aggregate(self.n, self.p, self.q, images, params_dict, jax_jit=True)
        surface_signature = aggregate[-1][0][-1]
        # Flatten the aggregated output.
        surface_signature_flat = surface_signature.reshape((surface_signature.shape[0], -1))
        # Feed-forward network
        x = self.dense1(surface_signature_flat)
        x = nn.relu(x)
        logits = self.dense2(x)
        return logits

def cross_entropy_loss(logits, labels):
    one_hot = jax.nn.one_hot(labels, num_classes=10)
    loss = optax.softmax_cross_entropy(logits=logits, labels=one_hot)
    return jnp.mean(loss)

def compute_accuracy(logits, labels):
    predictions = jnp.argmax(logits, axis=-1)
    return jnp.mean(predictions == labels)

# Training step: computes gradients, applies them, and returns updated state and loss.
@jax.jit
def train_step(state, batch):
    images, labels = batch

    def loss_fn(params):
        logits = state.apply_fn({'params': params}, images)
        loss = cross_entropy_loss(logits, labels)
        return loss
        
    grad_fn = jax.value_and_grad(loss_fn)
    loss, grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)

    return state, loss

# Evaluation step (for both validation and testing)
@jax.jit
def eval_step(state, batch):
    images, labels = batch
    logits = state.apply_fn({'params': state.params}, images)
    loss = cross_entropy_loss(logits, labels)
    acc = compute_accuracy(logits, labels)
    return loss, acc

def load_mnist_data():
    """
    Load MNIST, combine train and test splits,
    and then split into 70% training, 10% validation, and 20% testing.
    """
    ds_all = tfds.load('mnist', split=['train', 'test'], as_supervised=True)
    # Combine train and test into one list
    ds_combined = ds_all[0].concatenate(ds_all[1])
    
    # Convert to numpy arrays.
    images = []
    labels = []
    for image, label in tfds.as_numpy(ds_combined):
        # Convert image to float32 and normalize to [0,1]
        images.append(np.squeeze(image.astype(np.float32) / 255.0))
        labels.append(label)
    images = np.array(images)  # shape: (70000, 28, 28)
    labels = np.array(labels)

    # Shuffle the data.
    permutation = np.random.permutation(len(images))
    images = images[permutation]
    labels = labels[permutation]

    # Compute split indices.
    total = len(images)
    train_end = int(0.7 * total)
    valid_end = int(0.8 * total)  # 70% train, 10% validation, 20% test

    train_images, train_labels = images[:train_end], labels[:train_end]
    valid_images, valid_labels = images[train_end:valid_end], labels[train_end:valid_end]
    test_images, test_labels = images[valid_end:], labels[valid_end:]

    return (train_images, train_labels), (valid_images, valid_labels), (test_images, test_labels)

# Create batched dataset generator.
def create_batches(images, labels, batch_size):
    n_samples = images.shape[0]
    for i in range(0, n_samples, batch_size):
        batch_images = images[i:i+batch_size]
        batch_labels = labels[i:i+batch_size]
        yield batch_images, batch_labels

def main(batch_size, learning_rate, num_epochs, n, p, q):
    # Load the MNIST data.
    (train_images, train_labels), (valid_images, valid_labels), (test_images, test_labels) = load_mnist_data()
    
    # Initialize the model.
    model = ClassificationModel(n=n, p=p, q=q)
    rng = jax.random.PRNGKey(0)
    # Use a dummy input to initialize parameters.
    dummy_input = jnp.ones((batch_size, train_images.shape[1], train_images.shape[2]), jnp.float32)
    params = model.init(rng, dummy_input)['params']

    # Create optimizer and training state.
    optimizer = optax.adam(learning_rate)
    state = train_state.TrainState.create(apply_fn=model.apply, params=params, tx=optimizer)

    # Training loop.
    for epoch in range(1, num_epochs + 1):
        epoch_train_losses = []
        logging.info(f"Starting epoch {epoch}/{num_epochs}")

        # Use tqdm for training batches.
        for batch in tqdm(list(create_batches(train_images, train_labels, batch_size)),
                          desc=f"Epoch {epoch} [Training]", leave=False):
            state, loss = train_step(state, batch)
            epoch_train_losses.append(loss)

        train_loss = np.mean(jax.device_get(epoch_train_losses))
        logging.info(f"Epoch {epoch} | Train Loss: {train_loss:.4f}")

        # Validation loop.
        epoch_valid_losses = []
        epoch_valid_accs = []
        for batch in tqdm(list(create_batches(valid_images, valid_labels, batch_size)),
                          desc=f"Epoch {epoch} [Validation]", leave=False):
            loss, acc = eval_step(state, batch)
            epoch_valid_losses.append(loss)
            epoch_valid_accs.append(acc)
        valid_loss = np.mean(jax.device_get(epoch_valid_losses))
        valid_acc = np.mean(jax.device_get(epoch_valid_accs))
        logging.info(f"Epoch {epoch} | Validation Loss: {valid_loss:.4f} | Validation Acc: {valid_acc:.4f}")

    # Testing
    test_losses = []
    test_accs = []
    for batch in create_batches(test_images, test_labels, batch_size):
        loss, acc = eval_step(state, batch)
        test_losses.append(loss)
        test_accs.append(acc)
    test_loss = np.mean(jax.device_get(test_losses))
    test_acc = np.mean(jax.device_get(test_accs))
    logging.info(f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f}")

if __name__ == '__main__':
    main(batch_size=64, learning_rate=0.01, num_epochs=10, n=5, p=3, q=3)
