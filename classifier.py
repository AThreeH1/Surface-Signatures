from imports import *
from aggregate_torch import horizontal_first_aggregate
from custom_matrix_torch import mapping

class MNISTClassifier(pl.LightningModule):
    def __init__(self, input_size=9, num_classes=10):
        super(MNISTClassifier, self).__init__()
        
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
        Images = mapping(images.squeeze(1))
        aggregated_images = horizontal_first_aggregate(Images)  # Batch processing
        
        # Flatten aggregated output (9 elements per image in the batch)
        x = aggregated_images[0,0].value.matrix.view(9, -1)
        
        # Forward pass
        logits = self(x.T)
        loss = F.cross_entropy(logits, labels)
        
        # Accuracy
        acc = self.accuracy(logits, labels)
        self.log('train_loss', loss)
        self.log('train_acc', acc)
        
        return loss

    # def validation_step(self, batch, batch_idx):
    #     images, labels = batch
    #     Images = mapping(images.squeeze(1))
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
        Images = mapping(images.squeeze(1))
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

    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=7, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=7, shuffle=False)

    return train_loader, test_loader


if __name__ == "__main__":
    # Initialize DataLoader
    train_loader, test_loader = load_data(batch_size=640)
    
    # Initialize model
    model = MNISTClassifier()

    # Compile the core model for optimization
    model.model = torch.compile(model.model)

    # Set up PyTorch Lightning trainer
    logger = TensorBoardLogger('tb_logs', name='mnist_classifier')
    trainer = pl.Trainer(max_epochs=10, devices=1, accelerator='gpu', logger=logger, enable_progress_bar=True)

    # Train the model
    trainer.fit(model, train_loader)

    # Test the model
    test_results = trainer.test(model, test_loader)
    print(f"Test Results: {test_results}")

