from utils import *

def train():

    net = models.resnet18(pretrained=True)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3)

    num_ftrs = net.fc.in_features
    net.fc = nn.Linear(num_ftrs, 2)
    net.fc = net.fc
    net = nn.DataParallel(net).to(device)

    n_epochs = 35
    print_every = 10
    valid_loss_min = np.Inf
    val_loss = []
    val_acc = []
    train_loss = []
    train_acc = []
    total_step = len(train_loader)
    for epoch in range(1, n_epochs+1):
        running_loss = 0.0
        correct = 0
        total = 0
        print(f'Epoch {epoch},', 'lr: %.0e\n' % optimizer.param_groups[0]['lr'])
        for batch_idx, (data_, target_) in enumerate(train_loader):
            data_, target_ = data_.to(device, dtype=torch.float), target_.to(device, dtype=torch.long)
            optimizer.zero_grad()

            outputs = net(data_)
            loss = criterion(outputs, target_)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _,pred = torch.max(outputs, dim=1)
            correct += torch.sum(pred==target_).item()
            total += target_.size(0)
            if (batch_idx) % 20 == 0:
                print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
                       .format(epoch, n_epochs, batch_idx, total_step, loss.item()))
        train_acc.append(100 * correct / total)
        train_loss.append(running_loss/total_step)
        print(f'\ntrain-loss: {np.mean(train_loss):.4f}, train-acc: {(100 * correct/total):.4f}')

        batch_loss = 0
        total_t=0
        correct_t=0
        with torch.no_grad():
            net.eval()
            for data_t, target_t in (test_loader):
                data_t, target_t = data_t.to(device, dtype=torch.float), target_t.to(device, dtype=torch.long)
                outputs_t = net(data_t)
                loss_t = criterion(outputs_t, target_t)
                batch_loss += loss_t.item()
                _,pred_t = torch.max(outputs_t, dim=1)
                correct_t += torch.sum(pred_t==target_t).item()
                total_t += target_t.size(0)
            val_acc.append(100 * correct_t/total_t)
            val_loss.append(batch_loss/len(test_loader))
            network_learned = batch_loss < valid_loss_min
            print(f'validation loss: {np.mean(val_loss):.4f}, validation acc: {(100 * correct_t/total_t):.4f}\n')

            if network_learned:
                valid_loss_min = batch_loss
                torch.save(net.state_dict(), 'resnet18.pt')
                print('Improvement-Detected, save-model')
        scheduler.step(val_loss[-1])
        net.train()