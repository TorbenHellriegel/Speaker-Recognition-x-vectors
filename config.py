class Config:
    def __init__(self,
                batch_size=512,
                input_size=24,
                hidden_size=512,
                num_classes=1211, #TODO num classes of the training set or also the test set? 1211 or 1251 find out after full test run
                learning_rate=0.001,
                num_epochs=5,
                model_path='./trained_models/x_vector_model.pth',
                data_folder_path='../../../../../../../../../data/7hellrie'):
                
        self.data_folder_path = data_folder_path
        self.batch_size = batch_size
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.model_path = model_path
