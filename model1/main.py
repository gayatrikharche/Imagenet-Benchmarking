import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
import torch.optim as optim
from torch.utils.data import DataLoader
import os

def load_data_chunk(data_dir, chunk_file):
    # Unzip the chunk file
    os.system(f'unzip -q {chunk_file} -d {data_dir}')
    
    # Load the dataset
    transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    dataset = torchvision.datasets.ImageFolder(root=data_dir, transform=transform)
    return dataset

def main():
    # Directories and chunks
    data_dir = 'data'
    chunk_files = [
    'chunks/ILSVRC.zip.part0',
    'chunks/ILSVRC.zip.part1',
    'chunks/ILSVRC.zip.part2',
    'chunks/ILSVRC.zip.part3',
    'chunks/ILSVRC.zip.part4',
    'chunks/ILSVRC.zip.part5',
    'chunks/ILSVRC.zip.part6',
    'chunks/ILSVRC.zip.part7',
    'chunks/ILSVRC.zip.part8',
    'chunks/ILSVRC.zip.part9',
    'chunks/ILSVRC.zip.part10',
    'chunks/ILSVRC.zip.part11',
    'chunks/ILSVRC.zip.part12',
    'chunks/ILSVRC.zip.part13',
    'chunks/ILSVRC.zip.part14',
    'chunks/ILSVRC.zip.part15',
    'chunks/ILSVRC.zip.part16',
    'chunks/ILSVRC.zip.part17',
    'chunks/ILSVRC.zip.part18',
    'chunks/ILSVRC.zip.part19',
    'chunks/ILSVRC.zip.part20',
    'chunks/ILSVRC.zip.part21',
    'chunks/ILSVRC.zip.part22',
    'chunks/ILSVRC.zip.part23',
    'chunks/ILSVRC.zip.part24',
    'chunks/ILSVRC.zip.part25',
    'chunks/ILSVRC.zip.part26',
    'chunks/ILSVRC.zip.part27',
    'chunks/ILSVRC.zip.part28',
    'chunks/ILSVRC.zip.part29',
    'chunks/ILSVRC.zip.part30',
    'chunks/ILSVRC.zip.part31']
    
    # Initialize the model
    model = models.resnet50(pretrained=False)
    model = torch.nn.DataParallel(model)
    model.cuda()
    
    # Loss function and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
    
    # Training loop
    num_epochs = 20
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        
        for chunk_file in chunk_files:
            dataset = load_data_chunk(data_dir, chunk_file)
            train_loader = DataLoader(dataset, batch_size=256, shuffle=True, num_workers=4)
            
            for i, (inputs, labels) in enumerate(train_loader):
                inputs, labels = inputs.cuda(), labels.cuda()
                
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
                if i % 100 == 99:
                    print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {running_loss / 100:.4f}')
                    running_loss = 0.0
        
        # Save model after each epoch
        torch.save(model.state_dict(), f'model_epoch_{epoch}.pth')

    print('Training finished.')

if __name__ == '__main__':
    main()
