import torch


def train(train_loader, model, optimizer, criterion, device):
    """
    Trains network for one epoch in batches.

    Args:
        train_loader: Data loader for training set.
        model: Neural network model.
        optimizer: Optimizer (e.g. SGD).
        criterion: Loss function (e.g. cross-entropy loss).
    """
  
    avg_loss = 0
    correct = 0
    total = 0

    # Iterate through batches
    for i, data in enumerate(train_loader):
        # Get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

        # Move data to target device
        inputs, labels = inputs.to(device), labels.to(device)

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward + backward + optimize
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # Keep track of loss and accuracy
        avg_loss += loss
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    return avg_loss / len(train_loader), 100 * correct / total

def test(test_loader, model, criterion, device):
    """
    Evaluates network in batches.

    Args:
        test_loader: Data loader for test set.
        model: Neural network model.
        criterion: Loss function (e.g. cross-entropy loss).
    """

    avg_loss = 0
    correct = 0
    total = 0
    
    # Use torch.no_grad to skip gradient calculation, not needed for evaluation
    with torch.no_grad():
        # Iterate through batches
        for data in test_loader:
            # Get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # Move data to target device
            inputs, labels = inputs.to(device), labels.to(device)

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Keep track of loss and accuracy
            avg_loss += loss
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return avg_loss / len(test_loader), 100 * correct / total

def run(rnn_type, epochs=100, hidden_size=10):
    """
    Run a test on MNIST-1D

    Args:
        rnn_type: lstm
        epochs: number of epochs to run
        hidden_size: dimension of hidden state of rnn cell
    """
    # Create a writer to write to Tensorboard
    writer = SummaryWriter()

    if rnn_type == 'lstm':
        rnn = LSTM(1, hidden_size)
    else:
        raise Error('Unknown RNN type: ' + rnn_type)

    # Create classifier model
    model = nn.Sequential(OrderedDict([
        ('reshape', nn.Unflatten(1, (40, 1))),
        ('rnn', rnn),
        ('flat', nn.Flatten()),
        ('classifier', nn.Linear(40 * hidden_size, 10))
    ]))

    # Create loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), 1e-2, weight_decay=1e-3)

    # Use GPU if available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = model.to(device)

    for epoch in tqdm(range(epochs)):
        # Train on data
        train_loss, train_acc = train(train_loader,
                                      model,
                                      optimizer,
                                      criterion,
                                      device)

        # Test on data
        test_loss, test_acc = test(test_loader,
                                   model,
                                   criterion,
                                   device)

        # Write metrics to Tensorboard
        writer.add_scalars('Loss', {
            'Train_{}'.format(rnn_type): train_loss,
            'Test_{}'.format(rnn_type): test_loss
        }, epoch)
        writer.add_scalars('Accuracy', {
            'Train_{}'.format(rnn_type): train_acc,
            'Test_{}'.format(rnn_type): test_acc
        }, epoch)


    print('\nFinished.')
    writer.flush()
    writer.close()