class Config:
    def __init__(self, batch_size=100, input_size=24, hidden_size=512, num_classes=10, learning_rate=0.001,
                num_epochs=5, model_path='./trained_models/x_vector_model.pth', load_existing_model=True):
        self.batch_size = batch_size
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.model_path = model_path
        self.load_existing_model = load_existing_model