class Config:
    def __init__(self,
                batch_size=100,
                input_size=24,
                hidden_size=512,
                num_classes=10,
                learning_rate=0.001,
                num_epochs=5,
                load_existing_x_model=True,
                load_existing_plda_model=True,
                x_model_path='./trained_models/x_vector_model.pth',
                plda_model_path='./trained_models/plda_classifier_model.pth',
                data_folder_path='../../../../../../../../../data/7hellrie'):
                
        self.data_folder_path = data_folder_path
        # X-vector model parameters
        self.batch_size = batch_size
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.load_existing_x_model = load_existing_x_model
        self.x_model_path = x_model_path
        # PLDA model parameters
        self.load_existing_plda_model = load_existing_plda_model
        self.plda_model_path = plda_model_path