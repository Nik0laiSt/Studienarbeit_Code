#  FlowTransformer 2023 by liamdm / liam@riftcs.com
from framework.base_sequential import BaseSequential
from implementations.transformers.basic.decoder_block import TransformerDecoderBlock
from implementations.transformers.basic.encoder_block import TransformerEncoderBlock


class GPTSmallTransformer(BaseSequential):

    @property
    def name(self) -> str:
        return "GPT Model"

    @property
    def parameters(self) -> dict:
        return {
            "n_layers": self.n_layers,
            "internal_size": self.internal_size,
            "n_heads": self.n_heads,
            "dropout_rate": self.dropout_rate,
            "head_size": self.head_size
        }

    def __init__(self):
        super().__init__()
        self.n_layers = 12
        self.internal_size = 768
        self.n_heads = 12
        self.head_size = self.internal_size / self.n_heads
        self.dropout_rate = 0.02
        self.is_decoder = True

    def apply(self, X, prefix: str = None):
        #window_size = self.sequence_length
        real_size = X.shape[-1]

        m_x = X

        for layer_i in range(self.n_layers):
            m_x = TransformerDecoderBlock(real_size, self.internal_size, self.n_heads, dropout_rate=self.dropout_rate)(m_x)

        return m_x


class BERTSmallTransformer(BaseSequential):

    @property
    def name(self) -> str:
        return "BERT Model"

    @property
    def parameters(self) -> dict:
        return {
            "n_layers": self.n_layers,
            "internal_size": self.internal_size,
            "n_heads": self.n_heads,
            "dropout_rate": self.dropout_rate,
            "head_size": self.head_size
        }

    def __init__(self):
        super().__init__()
        self.n_layers = 12
        self.internal_size = 768
        self.n_heads = 12
        self.head_size = self.internal_size / self.n_heads
        self.dropout_rate = 0.02
        self.is_decoder = False

    def apply(self, X, prefix: str = None):
        #window_size = self.sequence_length
        real_size = X.shape[-1]

        m_x = X

        for layer_i in range(self.n_layers):
            m_x = TransformerEncoderBlock(real_size, self.internal_size, self.n_heads, dropout_rate=self.dropout_rate, prefix=f"block_{layer_i}_")(m_x, training=True)

        return m_x


from transformers import TFBertModel, BertConfig

class TinyBERTTransformer(BaseSequential):
    def __init__(self):
        super().__init__()
        # TinyBERT4-Konfiguration laden (TensorFlow-Version)
        self.bert_model = TFBertModel.from_pretrained(
            "huawei-noah/TinyBERT_General_4L_312D",
            from_pt=True  # lädt PyTorch-Gewichte in TensorFlow
        )

        # Konfigurationsparameter übernehmen
        config = self.bert_model.config
        self.n_layers = config.num_hidden_layers      # 4
        self.internal_size = config.hidden_size       # 312
        self.n_heads = config.num_attention_heads     # 12
        self.dropout_rate = config.hidden_dropout_prob
        self.head_size = self.internal_size // self.n_heads
        self.is_decoder = False

    def apply(self, X, prefix: str = None):
        # X: Tensor mit Input-IDs und optional Attention-Mask
        outputs = self.bert_model(X)
        # Gibt die letzte Hidden-State-Matrix zurück
        return outputs.last_hidden_state
