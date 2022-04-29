from tensorflow import keras

LR_SCHEDULE = [
    #(epoch to start, learning rate)
    (0, 0.01),
    (75, 0.001),
    (105, 0.0001),
]

#function to retrieve the scheduled learning rate based on epoch
def lr_scheduler(epoch, lr):
    learning_rate = lr
    for i in range(len(LR_SCHEDULE)):
        if epoch == LR_SCHEDULE[i][0]:
            learning_rate = LR_SCHEDULE[i][1]
    return learning_rate

#Callback function to set learning rate
class Custom_LearningRate_Scheduler(keras.callbacks.Callback):
    def __init__(self, schedule):
        super().__init__()
        self.schedule = schedule

    def on_epoch_begin(self, epoch, logs=None):
        if not hasattr(self.model.optimizer, 'lr'):
            raise ValueError('Optimizer must have a "lr" attribute')
        #Get current learning rate of model's optimizer
        lr = float(keras.backend.get_value(self.model.optimizer.learning_rate))
        #Call schedule function to get scheduled learning rate
        scheduled_lr = self.schedule(epoch, lr)
        #Set the value back to the optimizer before epoch starts
        keras.backend.set_value(self.model.optimizer.lr, scheduled_lr)
        print("\n Epoch {:d}: Learning rate is {:6.4f}.\n".format(epoch+1, scheduled_lr))

    