from ....models.wav2vec2.modeling_wav2vec2 import (
    WAV_2_VEC_2_INPUTS_DOCSTRING, 
    WAV_2_VEC_2_START_DOCSTRING, 
    Wav2Vec2Model,
    Wav2Vec2PreTrainedModel,
)
from ....utils import add_start_docstrings, add_start_docstrings_to_model_forward
from ...context import AdapterSetup
from ...heads import (
    AudioClassificationHead,
    CTCHead,
    ModelWithFlexibleHeadsAdaptersMixin,
)


@add_start_docstrings(
    """Wav2Vec2 Model transformer with the option to add multiple flexible heads on top.""",
    WAV_2_VEC_2_START_DOCSTRING,
)
class Wav2Vec2AdapterModel(ModelWithFlexibleHeadsAdaptersMixin, Wav2Vec2PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.wav2vec2 = Wav2Vec2Model(config)

        self._init_head_modules()

        self.init_weights()

    def freeze_feature_extractor(self):
        """
        Calling this function will disable the gradient computation for the feature encoder so that its parameters will
        not be updated during training.
        """
        warnings.warn(
            "The method `freeze_feature_extractor` is deprecated and will be removed in Transformers v5."
            "Please use the equivalent `freeze_feature_encoder` method instead.",
            FutureWarning,
        )
        self.freeze_feature_encoder()

    def freeze_feature_encoder(self):
        """
        Calling this function will disable the gradient computation for the feature encoder so that its parameter will
        not be updated during training.
        """
        self.wav2vec2.feature_extractor._freeze_parameters()

    def freeze_base_model(self):
        """
        Calling this function will disable the gradient computation for the base model so that its parameters will not
        be updated during training. Only the classification head will be updated.
        """
        for param in self.wav2vec2.parameters():
            param.requires_grad = False

    @add_start_docstrings_to_model_forward(WAV_2_VEC_2_INPUTS_DOCSTRING)
    def forward(
        self,
        input_values,
        attention_mask=None,
        mask_time_indices=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        labels=None,
        head=None,
        **kwargs
    ):
        input_values = input_values.view(-1, input_values.size(-1))
        attention_mask = attention_mask.view(-1, attention_mask.size(-1)) if attention_mask is not None else None
        # mask_time_indices?

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        output_hidden_states = True if self.config.use_weighted_layer_sum else output_hidden_states

        wav2vec2_outputs = self.wav2vec2(
            input_values,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        if head or AdapterSetup.get_context_head_setup() or self.active_head:
            outputs = self.forward_head(
                wav2vec2_outputs,
                input_values=input_values,
                head_name=head,
                attention_mask=attention_mask,
                return_dict=return_dict,
                labels=labels,
                **kwargs
            )
        else:
            # in case no head is used just return the output of the base model (including pooler output)
            return wav2vec2_outputs

        return outputs

    head_types = {
        "ctc": CTCHead,
        "classification": AudioClassificationHead,
    }

    def add_classification_head(
        self,
        head_name,
        num_labels=2,
        activation_function="tanh",
        overwrite_ok=False,
        id2label=None,
    ):
        """
        Adds a sequence classification head on top of the model.

        Args:
            head_name (str): The name of the head.
            num_labels (int, optional): Number of classification labels. Defaults to 2.
            activation_function (str, optional): Activation function. Defaults to 'tanh'.
            overwrite_ok (bool, optional): Force overwrite if a head with the same name exists. Defaults to False.
        """

        head = AudioClassificationHead(
            self,
            head_name,
            num_labels,
            activation_function,
            id2label,
        )
        self.add_prediction_head(head, overwrite_ok)

    def add_ctc_head(
        self,
        head_name,
        overwrite_ok=False,
    ):
        """
        Adds a Connectionist Temporal Classification (CTC) head on top of the model.

        Args:
            head_name (str): The name of the head.
            overwrite_ok (bool, optional): Force overwrite if a head with the same name exists. Defaults to False.
        """

        head = CTCHead(
            self,
            head_name,
        )
        self.add_prediction_head(head, overwrite_ok)

