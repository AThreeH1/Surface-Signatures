from imports import *
from aggregate_torch import horizontal_first_aggregate
from custom_matrix_torch import to_custom_matrix
from gl0_and_gl1_torch import GL0Element, GL1Element

wandb.login(key='7e169996e30d15f253a5f1d92ef090f2f3b948c4')

class MNISTClassifier(pl.LightningModule):
    def __init__(self, from_vector, kernel_gl1, input_size=9, num_classes=10):
        super(MNISTClassifier, self).__init__()
        
        self.from_vector = from_vector
        self.kernel_gl1 = kernel_gl1
        self.model = nn.Sequential(
            nn.Linear(input_size, num_classes)
            # nn.Linear(64, 32)
            # nn.Linear(32, num_classes)
        )
        self.accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=10)

    def forward(self, x):
        x = F.relu(self.model(x))
        # x = F.relu(self.fc2(x))
        # x = self.fc3(x)
        return x

    def training_step(self, batch, batch_idx):
        images, labels = batch
        # Assuming horizontal_first_aggregate function is available and batch-compatible
        print("next")
        tensor_images = images.squeeze(1)
        Images = to_custom_matrix(tensor_images, self.from_vector, self.kernel_gl1)
        print("here here")
        aggregated_images = horizontal_first_aggregate(Images)  # Batch processing
        print("x")
        # Flatten aggregated output (9 elements per image in the batch)
        x = aggregated_images[0,0].value.matrix.view(9, -1)
        print("y")
        # Forward pass
        logits = self(x.T)
        loss = F.cross_entropy(logits, labels)
        
        # Accuracy
        acc = self.accuracy(logits, labels)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_acc', acc, on_step=True, on_epoch=True, prog_bar=True)
        
        return loss

    # def validation_step(self, batch, batch_idx):
    #     images, labels = batch
    #     Images = to_custom_matrix(images.squeeze(1), self.from_vector, self.kernel_gl1)
    #     aggregated_images = horizontal_first_aggregate(Images)  # Batch processing
        
    #     # Flatten aggregated output (9 elements per image in the batch)
    #     x = aggregated_images[0,0].value.matrix.view(9, -1)
        
    #     logits = self(x)
    #     loss = F.cross_entropy(logits, labels)
        
    #     acc = self.accuracy(logits, labels)
    #     self.log('val_loss', loss)
    #     self.log('val_acc', acc)
        
    #     return loss

    def test_step(self, batch, batch_idx):
        images, labels = batch
        # Assuming horizontal_first_aggregate function is available and batch-compatible
        Images = to_custom_matrix(images.squeeze(1), self.from_vector, self.kernel_gl1)
        aggregated_images = horizontal_first_aggregate(Images)  # Batch processing

        # Flatten aggregated output (9 elements per image in the batch)
        x = aggregated_images[0, 0].value.matrix.view(9, -1)

        # Forward pass
        logits = self(x.T)
        loss = F.cross_entropy(logits, labels)

        # Accuracy
        acc = self.accuracy(logits, labels)
        self.log("test_loss", loss, prog_bar=True)
        self.log("test_acc", acc, prog_bar=True)

        return {"test_loss": loss, "test_acc": acc}

    
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


def load_data(batch_size=640):
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    dataset = datasets.MNIST(root='data', train=True, download=True, transform=transform)
    
    # Perform 80:20 split
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=8, shuffle=True, pin_memory=True, prefetch_factor=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=8, shuffle=False, pin_memory=True, prefetch_factor=2)

    return train_loader, test_loader


if __name__ == "__main__":

    def from_vector(m, Xt, Xs):
        n, p, q = 2, 1, 1
        fV = torch.eye(n + p).repeat(m, 1, 1)
        fU = torch.eye(n + q).repeat(m, 1, 1)
        dX = Xs - Xt

        fV[:, 0, 0] = fU[:, 0, 0] = torch.exp(dX)
        fV[:, 1, 1] = fU[:, 1, 1] = torch.exp(dX ** 2)
        fV[:, 2, 0] = torch.sin(dX)
        fV[:, 2, 1] = dX ** 5
        fU[:, 0, 2] = dX ** 3
        fU[:, 1, 2] = 7 * dX
 
        return GL0Element(m, n, p, q, fV, fU)


    def kernel_gl1(p1, p2, p3, p4):
        return (p1+p3-p2-p4).unsqueeze(-1).unsqueeze(-1)

    # Initialize DataLoader
    train_loader, test_loader = load_data(batch_size=640)
    
    # Initialize model
    model = MNISTClassifier(from_vector, kernel_gl1)

    wandb_logger = WandbLogger(project='SurfaceSignature', log_model="all")
    wandb_logger.watch(model, log="all", log_freq= 10, log_graph=True)

    # Set up PyTorch Lightning trainer
    logger = TensorBoardLogger('tb_logs', name='mnist_classifier')
    trainer = pl.Trainer(
        max_epochs=10, 
        devices=1, 
        accelerator='gpu', 
        logger=wandb_logger, 
        enable_progress_bar=True,
        log_every_n_steps=1
    )

    # Train the model
    trainer.fit(model, train_loader)

    # Test the model
    test_results = trainer.test(model, test_loader)
    print(f"Test Results: {test_results}")

 