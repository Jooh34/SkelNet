from utils.visualization import WriterTensorboardX

class skel_trainer():
    def __init__(self, config):
        self.start_epoch = 1
        self.epochs = config.trainer.epochs
        # self.writer = WriterTensorboardX(config.trainer.checkpoint_dir, self.train_logger, config.visualization.tensorboardX)

    def train(self):
        print('start train')
        for epoch in range(self.start_epoch, self.epochs+1):
            result = self._train_epoch(epoch)
    
    def _train_epoch(self, epoch):
        pass

