from imports import *
from aggregate_torch import horizontal_first_aggregate
from custom_matrix_torch import mapping

class MNISTClassifier(pl.LightningModule):
    def __init__(self, input_size=9, num_classes=10):
        super(MNISTClassifier, self).__init__()
        
        self.fc1 = nn.Linear(input_size, num_classes)
        # self.fc2 = nn.Linear(64, 32)
        # self.fc3 = nn.Linear(32, num_classes)
        
        self.accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=10)

    def forward(self, x):
        x = F.relu(self.fc1(x.T))
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
        logits = self(x)
        loss = F.cross_entropy(logits, labels)
        
        # Accuracy
        acc = self.accuracy(logits, labels)
        self.log('train_loss', loss)
        self.log('train_acc', acc)
        
        return loss

    def validation_step(self, batch, batch_idx):
        images, labels = batch
        Images = mapping(images.squeeze(1))
        aggregated_images = horizontal_first_aggregate(Images)  # Batch processing
        
        # Flatten aggregated output (9 elements per image in the batch)
        x = aggregated_images[0,0].value.matrix.view(9, -1)
        
        logits = self(x)
        loss = F.cross_entropy(logits, labels)
        
        acc = self.accuracy(logits, labels)
        self.log('val_loss', loss)
        self.log('val_acc', acc)
        
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


def load_data(batch_size=64):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    val_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader


if __name__ == "__main__":
    # Initialize DataLoader
    train_loader, val_loader = load_data(batch_size=64)
    
    # Initialize model
    model = MNISTClassifier()

    # Set up PyTorch Lightning trainer
    logger = TensorBoardLogger('tb_logs', name='mnist_classifier')
    trainer = pl.Trainer(max_epochs=10, devices=1, accelerator='cpu', logger=logger, enable_progress_bar=True)

    # Train the model
    trainer.fit(model, train_loader, val_loader)

    # Apply torch.compile for optimization
    model = torch.compile(model)
