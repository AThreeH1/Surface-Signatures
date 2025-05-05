import os
import jax
import jax.numpy as jnp
import optax
from flax import linen as nn
from flax.training import train_state
from torchvision.datasets import MNIST, FashionMNIST
from torch.utils.data import DataLoader
import numpy as np
from functools import partial
from aggregate_jax import jax_scan_aggregate
import logging 
from tqdm import tqdm
# import wandb

jax.clear_caches()

# Set memory growth for GPU
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# # Hyperparameters
# config = {
#     'seed': 3,
#     'learning_rate': 1e-4,
#     'batch_size': 128,
#     'num_epochs': 15,
#     'n': 4,
#     'p': 3,
#     'q': 3,
#     'project_name': 'jax-fashion-mnist-aggregate'
# }

# # Initialize W&B run
# wandb.init(project=config['project_name'], config=config)

class ClassificationModel(nn.Module):
    n: int   # parameter for aggregate 
    p: int   # parameter for aggregate
    q: int   # parameter for aggregate 

    @nn.compact  
    def __call__(self, images):
        # Learnable parameters
        a = self.param('a', nn.initializers.ones, (self.n,))
        b = self.param('b', nn.initializers.ones, (self.n,))
        b_prime = self.param('b_prime', nn.initializers.ones, (self.n,))
        c = self.param('c', nn.initializers.ones, (self.n,))
        c_prime = self.param('c_prime', nn.initializers.ones, (self.n,))
        d = self.param('d', nn.initializers.ones, (self.n,))
        d_prime = self.param('d_prime', nn.initializers.ones, (self.n,))
        e = self.param('e', nn.initializers.ones, (self.q,))

        params_dict = {
            'a': a,
            'b': b,
            'b_prime': b_prime,
            'c': c,
            'c_prime': c_prime,
            'd': d,
            'd_prime': d_prime,
            'e': e,
        }
        # Get the aggregated representation.
        aggregate = jax_scan_aggregate(self.n, self.p, self.q, images, params_dict, jax_jit=True) #shape (last element): rows, cols, batch_size, np, nq
        surface_signature = aggregate[-1]
        batch_surface_signature = jnp.transpose(surface_signature, (2, 0, 1, 3, 4))
        # batch_surface_signature = images[..., None, None] #Batch_size, Rows, Cols, 1, 1
        pooled_surface_signature = jax.lax.reduce_window( 
            batch_surface_signature,   
            init_value=0.0,  
            computation=jax.lax.add,  
            window_dimensions=(1, 4, 4, 1, 1),  
            window_strides=(1, 4, 4, 1, 1),  
            padding="VALID"
        ) / 16.0
        # Flatten the aggregated output.
        surface_signature_flat = pooled_surface_signature.reshape((pooled_surface_signature.shape[0], -1))
        # Feed-forward network
        x = nn.Dense(256)(surface_signature_flat)
        x = nn.relu(x)
        logits = nn.Dense(10)(x)
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
def train_step(state, images, labels):

    def loss_fn(params):
        logits = state.apply_fn({'params': params}, images)
        loss = cross_entropy_loss(logits, labels)
        acc = compute_accuracy(logits, labels)
        return loss, logits
        
    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (_, logits), grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)
    metrics = {'loss': cross_entropy_loss(logits, labels), 'accuracy': compute_accuracy(logits, labels)}

    return state, metrics

# Evaluation step (for both validation and testing)
@jax.jit
def eval_step(state, images, labels):
    logits = state.apply_fn({'params': state.params}, images)
    loss = cross_entropy_loss(logits, labels)
    acc = compute_accuracy(logits, labels)
    metrics = {'loss': loss, 'accuracy': acc}
    return metrics

def train_one_epoch(state, dataloader, epoch):
    """Train for 1 epoch on the training set."""
    batch_metrics = []
    for imgs, labels in tqdm(dataloader, desc=f"Training epoch {epoch}"):
        state, metrics = train_step(state, imgs, labels)
        batch_metrics.append(metrics)

    # Aggregate the metrics
    batch_metrics_np = jax.device_get(batch_metrics)  
    epoch_metrics_np = {
        k: np.mean([metrics[k] for metrics in batch_metrics_np])
        for k in batch_metrics_np[0]
    }

    return state, epoch_metrics_np

def evaluate_model(state, test_loader):
    """Evaluate on the test set in batches."""
    all_metrics = []
    for images, labels in tqdm(test_loader, desc="Evaluating"):
        metrics = eval_step(state, images, labels)
        all_metrics.append(metrics)
    
    # Transfer metrics from device (if using accelerator) to CPU and aggregate.
    all_metrics_np = jax.device_get(all_metrics)
    avg_metrics = {
        k: np.mean([m[k] for m in all_metrics_np])
        for k in all_metrics_np[0]
    }
    avg_metrics = jax.tree_map(lambda x: x.item(), avg_metrics)
    return avg_metrics

def custom_transform(x):
    # A couple of modifications here compared to tutorial #3 since we're using a CNN
    # Input: (28, 28) uint8 [0, 255] torch.Tensor, Output: (28, 28, 1) float32 [0, 1] np array
    return np.array(x, dtype=np.float32) / 255.

def custom_collate_fn(batch):
    """Provides us with batches of numpy arrays and not PyTorch's tensors."""
    transposed_data = list(zip(*batch))

    labels = np.array(transposed_data[1])
    imgs = np.stack(transposed_data[0])

    return imgs, labels

mnist_img_size = (28, 28)
batch_size = 128

train_dataset = FashionMNIST(root='train_fashion_mnist', train=True, download=True, transform=custom_transform)
test_dataset = FashionMNIST(root='test_fashion_mnist', train=False, download=True, transform=custom_transform)

train_loader = DataLoader(train_dataset, batch_size, shuffle=True, collate_fn=custom_collate_fn, drop_last=True)
test_loader = DataLoader(test_dataset, batch_size, shuffle=False, collate_fn=custom_collate_fn, drop_last=True)

# optimization - loading the whole dataset into memory
train_images = jnp.array(train_dataset.data)
train_lbls = jnp.array(train_dataset.targets)

test_images = jnp.array(test_dataset.data)
test_lbls = jnp.array(test_dataset.targets)

def create_train_state(n, p, q, learning_rate):

    model = ClassificationModel(n=n, p=p, q=q)
    rng = jax.random.PRNGKey(0)
    # Use a dummy input to initialize parameters.
    dummy_input = jnp.ones((batch_size, mnist_img_size[0], mnist_img_size[1]), jnp.float32)
    params = model.init(rng, dummy_input)['params']

    # Create optimizer and training state.
    optimizer = optax.adam(learning_rate)
    state = train_state.TrainState.create(apply_fn=model.apply, params=params, tx=optimizer)

    # TrainState is a simple built-in wrapper class that makes things a bit cleaner
    return state

# Finally let's define the high-level training/val loops
seed = 3  
learning_rate = 0.0001
num_epochs = 20
n = 4
p = 3
q = 3

train_state = create_train_state(n, p, q, learning_rate)

for epoch in range(1, num_epochs + 1):
    train_state, train_metrics = train_one_epoch(train_state, train_loader, epoch)
    print(f"Train epoch: {epoch}, loss: {train_metrics['loss']}, accuracy: {train_metrics['accuracy'] * 100}")

    test_metrics = evaluate_model(train_state, test_loader)
    print(f"Test epoch: {epoch}, loss: {test_metrics['loss']}, accuracy: {test_metrics['accuracy'] * 100}")


# OVERFITTING TEST

# # Pick 10 samples from the original training dataset
# small_raw_images = [train_dataset[i][0] for i in range(10)]  # Applies transform
# small_labels = [train_dataset[i][1] for i in range(10)]

# # Convert to batched arrays (Numpy -> JAX)
# small_images = jnp.array(np.stack(small_raw_images))         # shape (10, 28, 28)
# small_labels = jnp.array(small_labels)                       # shape (10,)

# # Overfit loop with debug prints
# state = create_train_state(n, p, q, learning_rate)
# for step in range(40):
#     state, metrics = train_step(state, small_images, small_labels)
#     print(f"[Step {step:2d}] loss={metrics['loss']:.4f}, acc={metrics['accuracy']*100:.1f}%")


