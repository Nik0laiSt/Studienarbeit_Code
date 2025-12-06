#  FlowTransformer 2023 by liamdm / liam@riftcs.com
from framework.base_sequential import BaseSequential
try:
    from tensorflow._api.v2.v2 import keras
except ImportError:
    from tensorflow import keras
from transformers import TFAutoModel

class TinyBERT(BaseSequential):
    def __init__(self, model_name="prajjwal1/bert-tiny"): #Can be swapped out for other tiny bert variants
        super().__init__()
        self.model_name = model_name
        self.transformer = TFAutoModel.from_pretrained(self.model_name, from_pt=True)
        self.transformer.trainable = False # Freeze the weights by default

    @property
    def name(self) -> str:
        return f"TinyBERT (Hugging Face: {self.model_name})"

    @property
    def parameters(self) -> dict:
        return {
            "model_name": self.model_name
        }

    def apply(self, X, prefix: str = None):
        """
        Applies the TinyBERT transformer.

        Args:
            X: Input tensor (pre-computed embeddings).
            prefix: Optional prefix.

        Returns:
            The output from the TinyBERT model.
        """

        # When building the model symbolically, the output of the transformer is a dict of tensors.
        # We must select the desired output ('last_hidden_state') within the Lambda function.
        outputs = keras.layers.Lambda(
            lambda x: self.transformer(inputs_embeds=x, attention_mask=None)['last_hidden_state'],
            output_shape=lambda input_shape: input_shape
        )(X)

        # The 'outputs' variable is now the desired last_hidden_state tensor.
        return outputs

    def set_trainable(self, trainable: bool):
        """
        Enable or disable training for the TinyBERT model.
        """
        self.transformer.trainable = trainable
