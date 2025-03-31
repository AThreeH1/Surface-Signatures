from imports import *
from aggregate_jax import jax_scan_aggregate
from torchvision.datasets import MNIST

wandb.login(key='7e169996e30d15f253a5f1d92ef090f2f3b948c4')

class MNISTClassifier(pl.LightningModule):
    def __init__(self, n, p, q, jax_scan_aggregate, lr, batch_size):
        """
        Args:
            n, p, q: parameters for your JAX function.
            jax_params: parameters for the JAX aggregation (a, b, c, d).
            jax_scan_aggregate: external JAX function to compute aggregated outputs.
            lr: learning rate.
            batch_size: batch size for data loaders.
        """
        super().__init__()
        self.n = n
        self.p = p
        self.q = q
        self.jax_scan_aggregate = jax_scan_aggregate
        self.lr = lr
        self.batch_size = batch_size

        self.a = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))
        self.b = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))
        self.c = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))
        self.d = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))
        self.e = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))

        self.feature_size = int((n+p)*(n+q))
        
        # Define the feed forward network:
        self.ffn = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.feature_size, 16),
            nn.ReLU(),
            nn.Linear(16, 10)
        )

        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, x):
        """
        x: input batch of MNIST images (B, 1, 28, 28)
        """

        x = x.squeeze(1)  # Now shape (B, 28, 28)

        # Convert Torch tensor to NumPy array and then to JAX array
        x_np = x.cpu().numpy()
        x_jax = jax.device_put(jnp.array(x_np))
        
        params = {
            'a': jnp.array(self.a.item()),
            'b': jnp.array(self.b.item()),
            'c': jnp.array(self.c.item()),
            'd': jnp.array(self.d.item()),
            'e': jnp.array(self.e.item())
        }


        agg = self.jax_scan_aggregate(self.n, self.p, self.q, x_jax, params, jax_jit=True)
        surface_signature_jax = agg[-1][0][-1]
        feature_np = np.array(surface_signature_jax)
        surface_signature = torch.tensor(feature_np, dtype=torch.float32, device=x.device)
        
        # Pass the flattened feature through the feed forward network.
        logits = self.ffn(surface_signature)
        return logits

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self.forward(x)
        loss = self.loss_fn(logits, y)
        self.log('train_loss', loss, on_step=True, on_epoch=True)
        preds = torch.argmax(logits, dim=1)
        acc = (preds == y).float().mean()
        self.log("train_acc", acc, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self.forward(x)
        loss = self.loss_fn(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = (preds == y).float().mean()
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', acc, prog_bar=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self.forward(x)
        loss = self.loss_fn(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = (preds == y).float().mean()
        self.log("test_loss", loss, prog_bar=True)
        self.log("test_acc", acc, prog_bar=True)

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.lr)

    @staticmethod
    def prepare_data():
        # Download MNIST data (only executed on one process)
        MNIST(root='./data', train=True, download=True)
        MNIST(root='./data', train=False, download=True)

    def setup(self, stage=None):
        # Define transforms for MNIST
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.squeeze(0))
        ])
        # Load the full training dataset
        full_dataset = MNIST(root='./data', train=True, transform=transform)

        train_size = int(0.7 * len(full_dataset))
        val_size = int(0.1 * len(full_dataset))
        test_size = int(0.2 * len(full_dataset))
        self.mnist_train, self.mnist_val, self.mnist_test = random_split(full_dataset, [train_size, val_size, test_size])

    def train_dataloader(self):
        return DataLoader(self.mnist_train, batch_size=self.batch_size, shuffle=True, num_workers=4)

    def val_dataloader(self):
        return DataLoader(self.mnist_val, batch_size=self.batch_size, shuffle=False, num_workers=4)

    def test_dataloader(self):
        return DataLoader(self.mnist_test, batch_size=self.batch_size, shuffle=False, num_workers=4)


if __name__ == '__main__':

    # Instantiate the MNIST classifier.
    model = MNISTClassifier(
        n=3,
        p=2,
        q=2,
        jax_scan_aggregate=jax_scan_aggregate,
        lr=0.001,
        batch_size=64
    )

    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss', 
        dirpath='checkpoints/',  
        filename='best_model',  
        save_top_k=1,  
        mode='min',  
        save_weights_only=True,  
    )

    early_stopping_callback = EarlyStopping(
        monitor='val_acc',  
        patience=3,  
        mode='max',  
        verbose=True,  
    )

    trainer = pl.Trainer(
        max_epochs=10,
        accelerator="auto",
        log_every_n_steps=25,
        callbacks=[checkpoint_callback, early_stopping_callback]
    )

    trainer.fit(model)
    trainer.test(model)