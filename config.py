class Config:
    def __init__(self,
                batch_size=512,
                input_size=24,
                hidden_size=512,
                num_classes=1211,
                x_vector_size=512,
                x_vec_extract_layer=6,
                learning_rate=0.001,
                num_epochs=6,
                batch_norm=True,
                dropout_p=0.0,
                augmentations_per_sample=2,
                plda_rank_f=150,
                model_path='./trained_models/x_vector_model.pth',
                data_folder_path='../../../../../../../../../data/7hellrie'):
                
        self.batch_size = batch_size
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.x_vector_size = x_vector_size
        self.x_vec_extract_layer = x_vec_extract_layer
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.batch_norm = batch_norm
        self.dropout_p = dropout_p
        self.augmentations_per_sample = augmentations_per_sample
        self.plda_rank_f = plda_rank_f
        self.model_path = model_path
        self.data_folder_path = data_folder_path
