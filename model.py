import torch.nn as nn
from transformers import AutoModel, AutoConfig
from peft import LoraConfig, get_peft_model


class TransformerBinaryClassifierWithLoRA(nn.Module):
    """
    Transformer-based binary classifier for hate speech detection with LoRA adaptation.
    """

    def __init__(self, model_name, dropout=0.1, lora_r=8, lora_alpha=16, lora_dropout=0.1):
        super(TransformerBinaryClassifierWithLoRA, self).__init__()
        
        # Load pretrained model and configuration
        self.encoder = AutoModel.from_pretrained(model_name)
        config = AutoConfig.from_pretrained(model_name)
        hidden_size = config.hidden_size

        # Define LoRA configuration
        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=None,
            lora_dropout=lora_dropout,
            bias="none",
            task_type="SEQ_CLS",
        )
        
        # Apply LoRA to the encoder model
        self.encoder = get_peft_model(self.encoder, lora_config)

        # Define the classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 1),
        )

    def forward(self, input_ids, attention_mask=None, labels=None):
        # Pass only inputs supported by the base encoder.
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        
        # Extract the output corresponding to the [CLS] token (first token) for classification
        cls_output = outputs.last_hidden_state[:, 0, :]
        
        # Pass through the classifier head to get logits
        logits = self.classifier(cls_output)
    
        loss = None
        if labels is not None:
            labels = labels.view(-1, 1).float()  # Ensure labels are in correct shape
            loss_fn = nn.BCEWithLogitsLoss()    # Binary cross-entropy loss
            loss = loss_fn(logits, labels)      # Calculate loss
    
        return {'loss': loss, 'logits': logits}


    def freeze_base_layers(self):
        """
        Freeze the layers of the base model (encoder), keeping only LoRA parameters trainable.
        """
        for param in self.encoder.parameters():
            param.requires_grad = False

        for name, param in self.encoder.named_parameters():
            if "lora" in name:
                param.requires_grad = True

        # Count frozen and trainable parameters
        frozen_params = sum(p.numel() for p in self.encoder.parameters() if not p.requires_grad)
        total_params = sum(p.numel() for p in self.parameters())
        
        print(f"Frozen {frozen_params:,} parameters out of {total_params:,} total parameters")
        print(f"Trainable parameters: {total_params - frozen_params:,}")

    def unfreeze_base_layers(self):
        """
        Unfreeze all layers of the base model (encoder), making all parameters trainable.
        """
        for param in self.encoder.parameters():
            param.requires_grad = True

        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"All parameters unfrozen. Trainable parameters: {trainable_params:,}")
