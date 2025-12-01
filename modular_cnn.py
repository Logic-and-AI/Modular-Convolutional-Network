import pandas as pd
import numpy as np
import torch
from sklearn import preprocessing
from sklearn import model_selection
from sklearn import linear_model
from sklearn import metrics
from sklearn import impute
import torch.optim as optim
import matplotlib.pyplot as plt
import torch.optim.lr_scheduler as lr_scheduler
import seaborn as sns
from sklearn.metrics import precision_recall_fscore_support

def train_model(model, train_loader, val_loader, optimizer, criterion, use_scheduler=False, include_eval=False, num_epochs=10, early_stop_threshold=90):
    """
    Συνάρτηση εκπαίδευσης, δέχεται τα εξής ορίσματα:
    1)  model                 : το μοντέλο προς εκπαίδευση
    2)  train_loader          : train set dataloader
    3)  val_loader            : validation set dataloader
    4)  optimizer             : optimizer για εκπαίδευση
    5)  criterion             : loss function
    6)  use_scheduler         : boolean παράμετρος, αν True τότε γίνεται χρήση του scheduler ReduceLROnPlateau, με learning rate factor = 0.5 και patience = 3
    7)  include_eval          : boolean παράμετρος, αν True τότε γίνεται call της συνάρτησης evaluate_model, επί των δεδομένων του val_loader, στο τέλος κάθε εποχής
    8)  num_epochs            : αριθμός εποχών εκπαίδευσης
    9)  early_stop_threshold  : όριο accuracy για early stopping, an include_eval=True, τότε αν το accuracy στo dataset του val_loader ξεπεράσει αυτό το όριο γίνεται
                                break στο loop της εκπαίδευσης (εναλλακτικά, αυτό θα μπορούσε να υλοποιηθεί όχι βάσει accuracy, αλλά διαφοράς στο loss ως προς train
                                και val datasets, προκειμένου η εκπαίδευση να διακόπτεται όταν το μοντέλο αρχίζει να εμφανίζει έντονα σημάδια overfitting).

    """
    model.to(device)
    train_loss_log = []
    val_acc_log = []
    val_loss_log = []
    scheduler=None

    if use_scheduler and include_eval: # έχω βάλει το scheduler να κάνει step βάσει val accuracy, όταν include_eval=True
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='max',
        factor=0.5,
        patience=3)
    elif use_scheduler and include_eval==False:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=3)

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            output = model(x_batch)
            loss = criterion(output, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            print(total_loss)


        avg_train_loss = total_loss / len(train_loader)
        train_loss_log.append(avg_train_loss)
        print(f"Epoch {epoch+1}, Train Loss: {avg_train_loss:.4f}")

        if include_eval:
            accuracy, eval_loss = evaluate_model(model, val_loader, criterion)
            val_acc_log.append(accuracy)
            val_loss_log.append(eval_loss)

            if accuracy >= early_stop_threshold:
                print(f"Early stop @ epoch {epoch+1}")
                break

        if use_scheduler and include_eval:
            scheduler.step(accuracy)
        elif use_scheduler and include_eval == False:
            scheduler.step(avg_train_loss)

    # plotting μετρικών στο τέλος της εκπαίδευσης:
    epochs = range(1, len(train_loss_log) + 1)
    fig, ax1 = plt.subplots()

    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss', color='tab:red')
    ax1.plot(epochs, train_loss_log, label='Train Loss', color='tab:red')
    if include_eval:
        ax1.plot(epochs, val_loss_log, label='Val Loss', color='tab:cyan')
    ax1.legend(loc='upper left')

    ax1.tick_params(axis='y', labelcolor='tab:red')

    if include_eval:
        ax2 = ax1.twinx()
        ax2.set_ylabel('Val Accuracy (%)', color='tab:blue')
        ax2.plot(epochs, val_acc_log, label='Val Accuracy', color='tab:blue')
        ax2.tick_params(axis='y', labelcolor='tab:blue')
        ax2.legend(loc='upper right')

    plt.title("Loss / Accuracy Plot:")
    fig.tight_layout()
    plt.show()




def evaluate_model(model, dataloader, criterion):
    """
    Συνάρτηση evaluation, δέχεται τα εξής ορίσματα:
    1) model      : το μοντέλο προς αξιολόγηση
    2) dataloader : το dataloader που θα χρησιμοποιηθεί για την αξιολόγηση
    3) criterion  : το loss function

    Επιστρέφει εκτίμηση του accuracy και του loss, στο dataset που αντιστοιχεί στο παρεχόμενο dataloader
    """
    model.eval()
    correct = 0
    total = 0
    total_loss = 0.0
    with torch.no_grad():
       for images, labels in dataloader:
           images = images.to(device)
           labels = labels.to(device)
           outputs = model(images)
           _, predicted = torch.max(outputs, 1)
           total += labels.size(0)
           correct += (predicted == labels).sum().item()
           loss = criterion(outputs, labels)
           total_loss += loss.item()

       accuracy = 100 * correct / total
       avg_val_loss = total_loss / len(dataloader)
       print(' -----------------------------')
       print(f"Evaluation of Accuracy: {accuracy:.2f}%")
       print(f"Evaluation of Loss: {avg_val_loss:.4f}")
       print(' -----------------------------')
       return accuracy, avg_val_loss


import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    """
    Κλάση για Residual Blocks. Blocks δηλαδή που περιλαμβάνουν και residual connections, με την αρχική είσοδο που δέχεται το block να προστίθεται στα
    feature maps που προκύπτουν έπειτα από έναν συγκεκριμένο αριθμό 2d convolutions. Εδώ, κάθε residual block περιλαμβάνει δύο κύρια convolutional layers.

    Ορίσματα:
    1) in_height      : η διάσταση ύψους της αρχικής εισόδου
    2) in_width       : η διάσταση πλάτους, τα δύο αρχικά ορίσματα χρησιμοποιούνται για τους υπολογισμούς διαστάσεων των διαφόρων layers
    3) kernel_size    : το μέγεθος του kernel για το πρώτο convolutional layer
    4) kernel_size_2  : το μέγεθος του kernel για το δεύτερο convolutional layer
    5) in_channels    : το πλήθος των καναλιών της εισόδου
    6) out_channels   : το πλήθος των καναλιών της εξόδου, το πρώτο conv2d layer είναι in_channels->out_channels, το δεύτερο conv2d layer είναι out_channels->out_channels
    7) batchnorm      : παράμετρος που συγκεκριμενοποιείται με int που είναι είτε το 1 είτε διάφορο του 1. Αν batchnorm=1 τότε κάθε κύριο conv2d ακολουθείται από batchnorm, διαφορετικά οχι.

    Το block έχει υλοποιηθεί με hard-coded stride=dilation=1 στα κύρια convolutional layers, κάτι που επιτρέπει, θέτοντας το κατάλληλο padding για το κάθε layer, να διατηρήσουμε τις διαστάσεις
    height, width της εξόδου ίδιες με αυτές της εισόδου.

    """
    def __init__(self, in_height, in_width, kernel_size, kernel_size_2, in_channels, out_channels, batchnorm):
        super().__init__()
        self.in_height = in_height
        self.in_width = in_width
        self.kernel_size = kernel_size
        self.kernel_size_2 = kernel_size_2
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.batchnorm = batchnorm
        self.padding1=(self.kernel_size - 1)//2 # υπολογισμός padding για το πρώτο conv2d layer (ώστε να μην έχουμε αλλαγή διαστάσεων)
        self.padding2=(self.kernel_size_2 -1)//2 #υπολογισμός padding για το δεύτερο layer

        if self.batchnorm == 1:
            self.conv1 = nn.Conv2d(in_channels=self.in_channels, out_channels=self.out_channels, kernel_size=self.kernel_size, stride=1, padding=self.padding1, dilation=1, bias=False)
            self.bn1 = nn.BatchNorm2d(self.out_channels)
            self.conv2 = nn.Conv2d(in_channels=self.out_channels, out_channels=self.out_channels, kernel_size=self.kernel_size_2, stride=1, padding=self.padding2, dilation=1, bias=False)
            self.bn2 = nn.BatchNorm2d(self.out_channels)
        else:
            self.conv1 = nn.Conv2d(in_channels=self.in_channels, out_channels=self.out_channels, kernel_size=self.kernel_size, stride=1, padding=self.padding1, dilation=1, bias=True)
            self.conv2 = nn.Conv2d(in_channels=self.out_channels, out_channels=self.out_channels, kernel_size=self.kernel_size_2, stride=1, padding=self.padding2, dilation=1, bias=True)

        if self.in_channels != self.out_channels: # αν ισχύει: χρησιμοποιούμε conv2d με kernel_size=1 προκειμένου αριθμός καναλιών εισόδου να είναι =out_channels, ώστε να μπορεί να προστεθεί με το τελικό feature map
            self.shortcut = nn.Conv2d(in_channels=self.in_channels, out_channels=self.out_channels, kernel_size=1, stride=1, bias=True)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
         identity = self.shortcut(x)
         if self.batchnorm == 1:
            out = F.relu(self.bn1(self.conv1(x)), inplace=True)
            out = self.bn2(self.conv2(out))
         else:
            out = F.relu(self.conv1(x), inplace=True)
            out = self.conv2(out)
         out += identity
         return F.relu(out, inplace=True)

    def output_shape(self):
        return self.in_height, self.in_width


class ConvolutionalBlock(nn.Module):
    """
    Απλό convolutional block. Σε αντίθεση με τον τρόπο που έχουν υλοποιηθεί τα residual blocks, τα convolutional blocks μπορεί να οδηγήσουν
    σε αλλαγή διαστάσεων ύψους, πλάτους.

    Ορίσματα:
    1) in_height    : η διάσταση ύψους της εισόδου στην μέθοδο forward
    2) in_width     : η διάσταση πλάτους της εισόδου
    3) kernel_size  : το μέγεθος του convolutional kernel
    4) in_channels  : το πλήθος καναλιών της εισόδου
    5) out_channels : το πλήθος καναλιών της εξόδου
    6) stride       : το stride (βηματισμός) των convolutions
    7) padding      : πόσο padding θα κάνουμε
    8) dilation     : dilation του kernel
    9) batchnorm    : όπως και στο residual block. Αν =1 τότε εφαρμόζεται batchnorm έπειτα από το convolution, διαφορετικά όχι

    """
    def __init__(self, in_height, in_width, kernel_size, in_channels, out_channels, stride, padding, dilation, batchnorm):
        super().__init__()
        self.padding = padding
        self.dilation = dilation
        self.in_height = in_height
        self.in_width = in_width
        self.kernel_size = kernel_size
        self.stride = stride
        self.batchnorm = batchnorm

        if self.batchnorm == 1:
            self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride=self.stride, padding=self.padding, dilation=self.dilation, bias=False)
            self.bn1 = nn.BatchNorm2d(out_channels)
        else:
            self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride=self.stride, padding=self.padding, dilation=self.dilation, bias=True)


    def forward(self, x):
      if self.batchnorm == 1:
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
      else:
        out = F.relu(self.conv1(x))
      return out


    def output_shape(self): # υπολογισμός διαστάσεων ύψους, πλάτους της εξόδου, βάσει των τύπων που αναφέρει και το documentation: https://docs.pytorch.org/docs/stable/generated/torch.nn.Conv2d.html#conv2d
        h1 = ((self.in_height+2*self.padding-self.dilation*(self.kernel_size-1)-1)//self.stride)+1
        w1 = ((self.in_width+2*self.padding-self.dilation*(self.kernel_size-1)-1)//self.stride)+1
        return h1, w1



class CNN(nn.Module):
    """
    Modular μέθοδος, που επιτρέπει να κάνουμε stack convolutional, residual και fully connected blocks. Convolutional και Residual blocks γίνονται
    append σε ένα nn.ModuleList, fully connected layers σε ένα άλλο. Βάσει των δύο ModuleLists ορίζονται στη συνέχεια αντίστοιχα nn.Sequential, σε κάθε
    ένα από τα οποία αντιστοιχεί ξεχωριστή forward μέθοδος. Συναρτήσει αυτών μπορεί στη συνέχεια να οριστεί εύκολα η forward μέθοδος της κλάσης CNN. Επιπλέον,
    το εκάστοτε block μπορεί να ακολουθείται από MaxPool2d Layer. Το αν αυτό θα συμβαίνει ή όχι καθορίζεται από τον τρόπο με τον οποίο έχουν παρασχεθεί
    τα ορίσματα της κλάσης.

    Ορίσματα:
    1) input_dim        : ο αριθμός καναλιών της εισόδου
    2) img_height       : η διάσταση ύψους της εισόδου
    3) img_width        : η διάσταση πλάτους της εισόδου
    4) block_specs      : μια λίστα από υπογραφές (λίστες που προσδιορίζουν επιμέρους χαρακτηριστικά του κάθε block), που περιέχει μία υπογραφή για το κάθε block.
                          Η υπογραφή για ένα residual block είναι μια λίστα από integers: [kernel_size, kernel_size_2, in_channels, out_channels, batchnorm],
                          όπου κάθε στοιχείο προσδιορίζει και το αντίστοιχο χαρακτηριστικό, κατά τρόπο αντίστοιχο όπως και στην σχετική κλάση.
                          Η υπογραφή για convolutional block, από την άλλη, είναι μια λίστα integers της μορφής: [kernel_size, in_channels, out_channels, stride, padding, dilation, batchnorm]
                          Βάσει του πλήθους των στοιχείων της εκάστοτε υπογραφής προσδιορίζεται και το αν αυτή αντιστοιχεί σε residual ή convolutional block.
                          Για παράδειγμα, block_specs=[[3, 3, 32, 1, 1, 1, 1], [3, 3, 32, 64, 1]] σημαίνει ότι έχουμε ένα convolutional block [3, 3, 32, 1, 1, 1, 1], δηλ kernel_size=3, in_channels=3,
                          out_channels=32, κτλ, καθώς κι ένα residual block [3, 3, 32, 64, 1], όπου kernel_size=3, kernel_size_2=3, in_channels=32 κτλ.
    5) pooling_kernel   : μία λίστα από integers, με len(pooling_kernel)=len(block_specs), η οποία προσδιορίζει το μέγεθος του kernel για το MaxPooling layer που ακολουθεί το εκάστοτε Block.
                          Αν η τιμή σε κάποια θέση της λίστας είναι 0, τότε το αντίστοιχο block δεν ακολουθείται από Pooling Layer.
    6) pooling_stride   : μία λίστα από integers, που προσδιορίζει το stride για το κάθε MaxPool Layer, len(pooling_stride)=len(pooling_kernel)
    7) pooling_padding  : ανάλογα, αλλά για το padding
    8) fc_layers        : μια λίστα από integers, που προσδιορίζει τον αριθμό νευρώνων για το κάθε layer (εκτός input) του fully connected network που αναλαμβάνει το τελικό classification.
                          Το τελευταίο μέλος της λίστα πρέπει να είναι ίσο με τον αριθμό των κλάσεων.
    9) fc_dropout       : Η πιθανότητα dropout για το fully connected network

    """
    def __init__(self, input_dim, img_height, img_width, block_specs,
                 pooling_kernel=[2, 2], pooling_stride=[2, 2], pooling_padding=[0, 0], fc_layers=[10], fc_dropout=0.0):
        super().__init__()
        self.conv_depth = len(block_specs)
        self.num_fc_layers = len(fc_layers)
        self.blocks = nn.ModuleList()
        self.fc_blocks = nn.ModuleList()
        self.fc_dropout = fc_dropout

        current_channels = input_dim
        current_in_height = img_height
        current_in_width = img_width
        if len(block_specs)==len(pooling_kernel) and len(pooling_kernel)==len(pooling_stride) and len(pooling_stride)==len(pooling_padding): #έλεγχος διαστάσεων
          for i in range(self.conv_depth):

            if len(block_specs[i]) == 5:
              block = ResidualBlock(in_height=current_in_height, in_width=current_in_width, kernel_size=block_specs[i][0], kernel_size_2=block_specs[i][1], in_channels=block_specs[i][2], out_channels=block_specs[i][3], batchnorm=block_specs[i][4])
              self.blocks.append(block)
              current_in_height, current_in_width = block.output_shape()
              if pooling_kernel[i] > 0: # αν η αντίστοιχη τιμή της λίστας pooling_kernel είναι >0, τότε το block ακολουθείται από MaxPool2d
                self.blocks.append(nn.MaxPool2d(kernel_size=pooling_kernel[i], stride=pooling_stride[i], padding=pooling_padding[i]))
                current_in_width = ((current_in_width + 2 * pooling_padding[i] - pooling_kernel[i]) // pooling_stride[i]) + 1 # υπολογισμός διάστασης πλάτους για την έξοδο του MaxPool2d
                current_in_height = ((current_in_height + 2 * pooling_padding[i] - pooling_kernel[i]) // pooling_stride[i]) + 1 # αντίστοιχα για το ύψος
              current_channels = block_specs[i][3]
            else:
              block = ConvolutionalBlock(in_height=current_in_height, in_width=current_in_width, kernel_size=block_specs[i][0], in_channels=block_specs[i][1], out_channels=block_specs[i][2], stride=block_specs[i][3], padding=block_specs[i][4], dilation=block_specs[i][5], batchnorm=block_specs[i][6])
              self.blocks.append(block)
              current_in_height, current_in_width = block.output_shape()
              if pooling_kernel[i] > 0:
                self.blocks.append(nn.MaxPool2d(kernel_size=pooling_kernel[i], stride=pooling_stride[i], padding=pooling_padding[i]))
                current_in_width = ((current_in_width + 2 * pooling_padding[i] - pooling_kernel[i]) // pooling_stride[i]) + 1
                current_in_height = ((current_in_height + 2 * pooling_padding[i] - pooling_kernel[i]) // pooling_stride[i]) + 1
              current_channels = block_specs[i][2]

          flattened_dim = current_in_height * current_in_width * current_channels # υπολογισμός διαστάσεων μετά το flatten, ώστε να καθοριστεί το μέγεθος του input layer για το fc network
          self.flattened_dim = flattened_dim
          if self.num_fc_layers>1:
              for i in range(self.num_fc_layers-1):
                self.fc_blocks.append(nn.Linear(self.flattened_dim, fc_layers[i]))
                self.fc_blocks.append(nn.ReLU())
                if self.fc_dropout > 0:
                    self.fc_blocks.append(nn.Dropout(p=self.fc_dropout))
                self.flattened_dim = fc_layers[i]
              self.fc_blocks.append(nn.Linear(self.flattened_dim, fc_layers[-1]))
          elif self.num_fc_layers==1:
              self.fc_blocks.append(nn.Linear(self.flattened_dim, fc_layers[0]))

          else:
              raise ValueError("fc_layers must be a list of length at least 1.")

        else:
            raise ValueError("All supplied lists for the convolutional layers must be of the same length.")


        self.block_sequence = nn.Sequential(*self.blocks)  # μετατροπή της nn.ModuleList λίστας σε Sequential, οπότε δεν χρειάζεται να εφαρμοστεί for loop στην αντίστοιχη forward μέθοδο. Ανάλογο παράδειγμα χρήσης του nn.Sequential, με *layers: https://github.com/bamos/densenet.pytorch/blob/master/densenet.py#L83
        self.fc_block_sequence = nn.Sequential(*self.fc_blocks)

    def _forward_convs(self, x): # forward μέθοδος για residual, convolutional portion
        return self.block_sequence(x)

    def _forward_fc(self, x): # forward μέθοδος για fc portion
        return self.fc_block_sequence(x)

    def forward(self, x):
        x = self._forward_convs(x)
        x = x.flatten(start_dim=1)
        x = self._forward_fc(x)
        return x






def confusion_matrix(model, dataloader, class_names=None, device='cuda'):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    num_classes = len(class_names) if class_names else len(set(all_labels))
    cm = confusion_matrix_helper(all_labels, all_preds, num_classes)

    if class_names is None:
        class_names = [f"Class {i}" for i in range(num_classes)]

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix Heatmap')
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()





def confusion_matrix_helper(y_true, y_pred, num_classes):

    cm = np.zeros((num_classes, num_classes), dtype=int)

    for true_label, pred_label in zip(y_true, y_pred):
        cm[true_label][pred_label] += 1

    return cm


def per_class_report(model, dataloader, class_names, device='cuda'):

    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    precision, recall, fscore, _ = precision_recall_fscore_support(
        all_labels, all_preds, labels=range(len(class_names)), zero_division=0
    )

    for i, class_name in enumerate(class_names):
        print(f"Class: {class_name}")
        print(f"Precision: {precision[i]}")
        print(f"Recall:    {recall[i]}")
        print(f"F-score:   {fscore[i]}")
        print()
