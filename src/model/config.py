"""
Model configuration for SpectraFormer
"""

class SpectraFormerConfig:
    """Configuration for SpectraFormer model"""
    
    # Input/Output
    input_dim = 331  # Number of spectral features
    num_classes = 24  # Number of crop varieties
    
    # Transformer
    hidden_dim = 128
    num_layers = 4
    num_heads = 8
    mlp_dim = 512
    dropout = 0.3
    
    # Training
    learning_rate = 0.001
    batch_size = 16
    epochs = 200
    weight_decay = 1e-5
    
    # Data split
    train_ratio = 0.8
    val_ratio = 0.1
    test_ratio = 0.1
    
    # Device
    device = "cuda"  # Will be overridden by get_device()
    
    def __str__(self):
        return f"""
SpectraFormer Config:
  Input Dim: {self.input_dim}
  Num Classes: {self.num_classes}
  Hidden Dim: {self.hidden_dim}
  Num Layers: {self.num_layers}
  Num Heads: {self.num_heads}
  Learning Rate: {self.learning_rate}
  Batch Size: {self.batch_size}
  Epochs: {self.epochs}
  Dropout: {self.dropout}
"""