import torch
import torch.nn as nn
from transformers import AutoModel, AutoModelForSequenceClassification

class TinyBertWithGAP(nn.Module):
    def __init__(self, model_name, num_labels, **kwargs):
        super(TinyBertWithGAP, self).__init__()
        self.num_labels = num_labels
        # Lade nur das Basis-BERT-Modell (ohne Klassifikationskopf)
        self.bert = AutoModel.from_pretrained(model_name, **kwargs)
        self.config = self.bert.config  # Get the config from the base model
        self.config.num_labels = num_labels # Explicitly set num_labels
        self.dropout = nn.Dropout(0.1)
        # Der neue Klassifikator ist nur ein linearer Layer
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, labels=None):
        # Pass all three inputs to the base BERT model
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids  # <-- Add this line
        )
        
        last_hidden_state = outputs.last_hidden_state 
        
        # === Global Average Pooling ===
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
        sum_mask = input_mask_expanded.sum(1)
        sum_mask = torch.clamp(sum_mask, min=1e-9) 
        pooled_output = sum_embeddings / sum_mask
        # =============================
        
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        return {
            'loss': loss, 
            'logits': logits, 
            'attentions': outputs.attentions
        }