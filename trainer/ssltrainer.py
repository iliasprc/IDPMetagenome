import os

import torch

from idp_methods.utils import *
from trainer.basetrainer import BaseTrainer
from trainer.util import MetricTracker, write_csv, save_model, make_dirs


class SSLTrainer(BaseTrainer):
    """
    Trainer class
    """

    def __init__(self, config, model, optimizer, data_loader, writer, checkpoint_dir, logger, class_dict,
                 valid_data_loader=None, test_data_loader=None, lr_scheduler=None, metric_ftns=None):
        super(SSLTrainer, self).__init__(config, data_loader, writer, checkpoint_dir, logger,
                                         valid_data_loader=valid_data_loader,
                                         test_data_loader=test_data_loader, metric_ftns=metric_ftns)

        if (self.config.cuda):
            use_cuda = torch.cuda.is_available()
            self.device = torch.device("cuda" if use_cuda else "cpu")
        else:
            self.device = torch.device("cpu")
        self.start_epoch = 1
        self.train_data_loader = data_loader

        self.len_epoch = self.config.batch_size * len(self.train_data_loader)
        self.epochs = self.config.epochs
        self.valid_data_loader = valid_data_loader
        self.test_data_loader = test_data_loader
        self.do_validation = self.valid_data_loader is not None
        self.do_test = self.test_data_loader is not None

        self.log_step = self.config.log_interval
        self.model = model
        self.num_classes = len(class_dict)
        self.optimizer = optimizer
        self.gradient_accumulation = config.gradient_accumulation
        from models.utils import Cosine_LR_Scheduler
        self.scheduler = Cosine_LR_Scheduler(
            self.optimizer,
            warmup_epochs=20, warmup_lr=0,
            num_epochs=self.epochs + 2, base_lr=self.config['model']['optimizer']['lr'], final_lr=1e-5,
            iter_per_epoch=len(self.train_data_loader) // self.gradient_accumulation,
            constant_predictor_lr=True  # see the end of section 4.2 predictor
        )
        self.mnt_best = np.inf
        # if self.config.dataset.type == 'multi_target':
        self.criterion = torch.nn.BCEWithLogitsLoss(reduction='mean')
        self.criterion = torch.nn.CrossEntropyLoss(reduction='mean')

        self.checkpoint_dir = checkpoint_dir

        self.writer = writer
        self.metric_ftns = ['loss', 'acc']
        self.train_metrics = MetricTracker(*[m for m in self.metric_ftns], writer=self.writer, mode='train')
        self.metric_ftns = ['loss', 'acc']
        self.valid_metrics = MetricTracker(*[m for m in self.metric_ftns], writer=self.writer, mode='validation')
        self.logger = logger

        self.confusion_matrix = torch.zeros(2, 2)

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch
        Args:
            epoch (int): current training epoch.
        """

        self.model.train()
        self.confusion_matrix = 0 * self.confusion_matrix
        self.train_metrics.reset()

        gradient_accumulation = self.gradient_accumulation
        for batch_idx, (data, target) in enumerate(self.train_data_loader):
            # print(data.shape,data[:,0:-1].shape)
            target = data[:, 1:].to(self.device)
            data = data[:, 0:-1].to(self.device)

            output = self.model(data)
            ###########################print(target.shape,output.shape)
            loss = self.criterion(output.squeeze(0), target.squeeze(0))
            loss = loss.mean()

            (loss / gradient_accumulation).backward()
            if (batch_idx % gradient_accumulation == 0):
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.scheduler.step()
                self.optimizer.step()  # Now we can do an optimizer step
                self.optimizer.zero_grad()  # Reset gradients tensors

            prediction = torch.max(output, 1)

            # print(output.shape)
            ol = output.detach().cpu().numpy().tolist()

            writer_step = (epoch - 1) * self.len_epoch + batch_idx

            self.train_metrics.update(key='loss', value=loss.item(), n=1, writer_step=writer_step)

            # for t, p in zip(target.cpu().view(-1), prediction[1].cpu().view(-1)):
            #     self.confusion_matrix[t.long(), p.long()] += 1
            self._progress(batch_idx, epoch, metrics=self.train_metrics, mode='train')

        # print_metrics(metrics, self.logger)
        self._progress(batch_idx, epoch, metrics=self.train_metrics, mode='train', print_summary=True)

    def _valid_epoch(self, epoch, mode, loader):
        """
        Args:
            epoch (int): current epoch
            mode (string): 'validation' or 'test'
            loader (dataloader):
        Returns: validation loss
        """
        self.model.eval()
        self.valid_sentences = []
        self.valid_metrics.reset()

        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(loader):
                target = data[:, 1:].to(self.device)
                data = data[:, 0:-1].to(self.device)
                output = self.model(data)
                loss = self.criterion(output.squeeze(0), target.squeeze(0))
                loss = loss.mean()
                writer_step = (epoch - 1) * len(loader) + batch_idx

                prediction = torch.max(output, 1)

                self.valid_metrics.update(key='loss', value=loss.item(), n=1, writer_step=writer_step)
        # pred = np.array(yhat)
        # target = np.array(y)

        self._progress(batch_idx, epoch, metrics=self.valid_metrics, mode=mode, print_summary=True)
        k = 5

        val_loss = self.valid_metrics.avg('loss')

        return val_loss

    def train(self):
        """
        Train the model
        """
        for epoch in range(self.start_epoch, self.epochs):
            # torch.manual_seed(self.config.seed)
            self._train_epoch(epoch)

            self.logger.info(f"{'!' * 10}    VALIDATION   , {'!' * 10}")
            validation_loss = self._valid_epoch(epoch, 'validation', self.valid_data_loader)
            make_dirs(self.checkpoint_dir)

            self.checkpointer(epoch, validation_loss)
            # self.lr_scheduler.step(validation_loss)
            if self.do_test:
                self.logger.info(f"{'!' * 10}    TEST  , {'!' * 10}")
                self._valid_epoch(epoch, 'test', self.test_data_loader)

    def predict(self, epoch):
        """
        Inference
        Args:
            epoch ():
        Returns:
        """
        self.model.eval()

        predictions = []
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(self.test_data_loader):
                data = data.to(self.device)

                logits = self.model(data, None)

                maxes, prediction = torch.max(logits, 1)  # get the index of the max log-probability
                # log.info()
                predictions.append(f"{target[0]},{prediction.cpu().numpy()[0]}")

        pred_name = os.path.join(self.checkpoint_dir, f'validation_predictions_epoch_{epoch:d}_.csv')
        write_csv(predictions, pred_name)
        return predictions

    def checkpointer(self, epoch, metric):

        is_best = metric < self.mnt_best
        if (is_best):
            self.mnt_best = metric

            self.logger.info(f"Best val loss {self.mnt_best} so far ")
            # else:
            #     self.gradient_accumulation = self.gradient_accumulation // 2
            #     if self.gradient_accumulation < 4:
            #         self.gradient_accumulation = 4

            save_model(self.checkpoint_dir, self.model, self.optimizer, self.valid_metrics.avg('loss'), epoch,
                       f'_model_best')
        save_model(self.checkpoint_dir, self.model, self.optimizer, self.valid_metrics.avg('loss'), epoch,
                   f'_model_last')

    def _progress(self, batch_idx, epoch, metrics, mode='', print_summary=False):
        metrics_string = metrics.calc_all_metrics()
        if ((batch_idx * self.config.batch_size) % self.log_step == 0):

            if metrics_string == None:
                self.logger.warning(f" No metrics")
            else:
                self.logger.info(
                    f"{mode} Epoch: [{epoch:2d}/{self.epochs:2d}]\t Sample ["
                    f"{batch_idx * self.config.batch_size:5d}/{self.len_epoch:5d}]\t {metrics_string}")
        elif print_summary:
            self.logger.info(
                f'{mode} summary  Epoch: [{epoch}/{self.epochs}]\t {metrics_string}')
