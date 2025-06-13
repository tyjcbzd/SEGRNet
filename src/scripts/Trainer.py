import time
from scripts.Evaluation import *
from utils.helpers import *
from utils.loss import lovasz_softmax


class Trainer(object):
    def __init__(self, model, optimizer, size, 
                 batch_size, train_log_path, 
                 train_loader, val_loader, 
                 device, checkpoint_path,
                 num_epoch, lr, loss_region, 
                 logger):
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.num_epoch = num_epoch
        self.iteration = 0
        self.train_log_path = train_log_path
        self.size = size
        self.batch_size = batch_size
        self.optimizer = optimizer
        self.model = model
        self.device = device
        self.checkpoint_path = checkpoint_path
        self.lr = lr
        self.loss_region = loss_region
        self.logger = logger

    def train(self):
        loss_name = "Lovas softmax"

        # print basic info
        data_str = f"Start Time: {time.time()}"
        data_str += f"Hyperparameters:\n Image Size: {self.size}\n"
        data_str += f"Batch Size: {self.batch_size}\nLR: {self.lr}\nEpochs: {self.num_epoch}\n"
        data_str += f"Optimizer: Adam\nLoss: {loss_name}\n"

        self.logger.add_b_info(data_str)
        
        self.logger.set_names(["Epoch", 'Jaccard', 'F1', 'Recall', 'Precision', 'Specificity', 'ACC', 'F2'])

        best_res = {'Jaccard':0, 'Dice':0, 'Recall':0, 'Precision':0, 'Specificity':0, 'Acc':0, 'F2':0}
        print("Start Training...")
        best_valid_loss = float('inf')

        for epoch in range(self.num_epoch):
            train_epoch_loss = self.train_epoch(epoch=epoch)

            # 每第10次计算损失
            if (epoch % 1) == 0:
                valid_epoch_loss, eva_epoch_res = self.evaluate(epoch)
                
                if valid_epoch_loss < best_valid_loss:
                    best_valid_loss = valid_epoch_loss
                    best_loss_path = self.checkpoint_path + 'best_loss_checkpoint222.pth'
                    print(f'Saving best loss checkpoints: {best_loss_path}')
                    torch.save(self.model.state_dict(), best_loss_path)

        return train_epoch_loss, valid_epoch_loss

    def train_epoch(self,epoch):

        print(f'======{epoch+1}: Start Training======== ')
        epoch_loss = 0
        self.model.train()
        for i, (image, target, edge) in enumerate(self.train_loader):
            self.optimizer.zero_grad()
            image, target, edge = image.to(self.device), target.to(self.device), edge.to(self.device)
            edge_pred, region_pred = self.model(image)

            edge_loss = lovasz_softmax(edge_pred, edge)
            region_loss = self.loss_region(region_pred,target)
            two_loss = edge_loss + region_loss
            two_loss.backward()
            epoch_loss += two_loss.item()
            self.optimizer.step()
        avg_train_loss = epoch_loss / len(self.train_loader)
        return avg_train_loss

    def evaluate(self, epoch):
        epoch_loss = 0
        self.model.eval()
        total_score = []
        metrics_score = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        with torch.no_grad():
            for i, (image, target, edge) in enumerate(self.val_loader):
                image, target, edge= image.to(self.device), target.to(self.device), edge.to(self.device)

                edge_pred, region_pred = self.model(image)
                y_pred = region_pred
                # _, dim, _, _ = region_pred.shape
                edge_loss = lovasz_softmax(edge_pred, edge)
                region_loss = self.loss_region(region_pred,target)
                two_loss = edge_loss + region_loss
                epoch_loss += two_loss.item()

                # for lovas——softmax
                # if dim == 2:
                #     region_pred = torch.argmax(region_pred, dim=1)
                #     region_pred = region_pred.unsqueeze(0)

                y_true = target.cpu().numpy()
                y_pred = y_pred.cpu().numpy()

                y_pred = y_pred > 0.5
                y_pred = y_pred.reshape(-1)
                y_pred = y_pred.astype(np.uint8)

                y_true = y_true > 0.5
                y_true = y_true.reshape(-1)
                y_true = y_true.astype(np.uint8)
                
                # compute metrics
                score_jaccard = miou_score(y_true, y_pred)
                score_f1 = dice_score(y_true, y_pred)
                score_recall = recall_score(y_true, y_pred)
                score_precision = precision_score(y_true, y_pred)
                score_fbeta = F2_score(y_true, y_pred)
                score_acc = accuracy_score(y_true, y_pred)
                score_specificity = specificity_score(y_true, y_pred)

        # results for each epoch
        epoch_res = {'Jaccard':score_jaccard, 'Dice':score_f1, 'Recall':score_recall,
                     'Precision':score_precision, 'Specificity':score_specificity, 'Acc':score_acc, 'F2':score_fbeta}
        
        # add into logger
        self.logger.append([int(epoch+1),score_jaccard,score_f1,score_recall,score_precision, score_specificity, score_acc,score_fbeta])
        epoch_loss = epoch_loss / len(self.val_loader)

        print(f"====={epoch+1}: Evaluate Results =======")
        print(epoch_res)

        return epoch_loss, epoch_res
