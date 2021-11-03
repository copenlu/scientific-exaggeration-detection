from torch import nn
from typing import AnyStr, List
from transformers import AutoConfig
from transformers import AutoModel


class AutoTransformerMultiTask(nn.Module):
    """
    Implements a transformer with multiple classifier heads for multi-task training
    """
    def __init__(self, transformer_model: AnyStr, task_num_labels: List):
        super(AutoTransformerMultiTask, self).__init__()

        config = AutoConfig.from_pretrained(transformer_model)
        self.config = config
        self.xformer = AutoModel.from_pretrained(transformer_model, config=config)

        # Pooling layers
        self.pooling = nn.ModuleList([nn.Linear(config.hidden_size, config.hidden_size) for _ in task_num_labels])
        self.act = nn.Tanh()

        # Create the classifier heads
        self.task_classifiers = nn.ModuleList([nn.Linear(config.hidden_size, n_labels) for n_labels in task_num_labels])
        self.task_num_labels = task_num_labels

        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            labels=None,
            task_num=0,
            lam=1.0,
            logits_mask=None
    ):
        outputs = self.xformer(
            input_ids,
            attention_mask=attention_mask
        )

        sequence_output = outputs[0]
        if len(sequence_output.shape) == 3:
            sequence_output = sequence_output[:,0]
        assert sequence_output.shape[0] == input_ids.shape[0]
        assert sequence_output.shape[1] == self.config.hidden_size

        pooled_output = self.pooling[task_num](sequence_output)
        pooled_output = self.dropout(self.act(pooled_output))

        logits = self.task_classifiers[task_num](pooled_output)

        outputs = (logits,)
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = lam * loss_fct(logits.view(-1, self.task_num_labels[task_num]), labels.view(-1))
            outputs = (loss,) + outputs

        return outputs


class AutoTransformerForSentenceSequenceModeling(nn.Module):
    """
       Implements a transformer which performs sequence classification on a sequence of sentences
    """

    def __init__(self, transformer_model: AnyStr, num_labels: int = 2, sep_token_id: int = 2):
        super(AutoTransformerForSentenceSequenceModeling, self).__init__()

        config = AutoConfig.from_pretrained(transformer_model)
        self.config = config
        self.xformer = AutoModel.from_pretrained(transformer_model, config=config)

        # Pooling layers
        self.pooling = nn.Linear(config.hidden_size, config.hidden_size)
        self.act = nn.Tanh()

        # Create the classifier heads
        self.classifier = nn.Linear(config.hidden_size, num_labels)
        self.num_labels = num_labels

        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.sep_token_id = sep_token_id

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            labels=None,
            lam=1.0
    ):
        outputs = self.xformer(
            input_ids,
            attention_mask=attention_mask,

        )

        # Gather all of the SEP hidden states VERIFY THIS IS CORRECT!
        hidden_states = outputs[0].reshape(-1, self.config.hidden_size)
        locs = (input_ids == self.sep_token_id).view(-1)
        #(n * seq_len x d) -> (n * sep_len x d)
        sequence_output = hidden_states[locs]
        assert sequence_output.shape[0] == sum(locs)
        assert sequence_output.shape[1] == self.config.hidden_size

        pooled_output = self.pooling(sequence_output)
        pooled_output = self.dropout(self.act(pooled_output))

        logits = self.classifier(pooled_output)

        outputs = (logits,)
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = lam * loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs

        return outputs


class AutoTransformerForSentenceSequenceModelingMultiTask(nn.Module):
    """
       Implements a transformer which performs sequence classification on a sequence of sentences
    """

    def __init__(self, transformer_model: AnyStr, task_num_labels: List, sep_token_id: int = 2):
        super(AutoTransformerForSentenceSequenceModelingMultiTask, self).__init__()

        config = AutoConfig.from_pretrained(transformer_model)
        self.config = config
        self.xformer = AutoModel.from_pretrained(transformer_model, config=config)

        # Pooling layers
        self.pooling = nn.ModuleList([nn.Linear(config.hidden_size, config.hidden_size) for _ in task_num_labels])
        self.act = nn.Tanh()

        # Create the classifier heads
        self.task_classifiers = nn.ModuleList([nn.Linear(config.hidden_size, n_labels) for n_labels in task_num_labels])
        self.task_num_labels = task_num_labels

        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.sep_token_id = sep_token_id

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            labels=None,
            logits_mask=None,
            task_num=0,
            lam=1.0
    ):
        outputs = self.xformer(
            input_ids,
            attention_mask=attention_mask,

        )

        # Gather all of the SEP hidden states VERIFY THIS IS CORRECT!
        #hidden_states = outputs[0].reshape(-1, self.config.hidden_size)
        # locs = (input_ids == self.sep_token_id).view(-1)
        # # (n * seq_len x d) -> (n * sep_len x d)
        # sequence_output = hidden_states[locs]
        if logits_mask is None:
            sequence_output = outputs[0][:,0,:].reshape(-1, self.config.hidden_size)
        else:
            sequence_output = outputs[0][logits_mask == 1].reshape(-1, self.config.hidden_size)
            assert sequence_output.shape[0] == logits_mask.sum().item()
        assert sequence_output.shape[1] == self.config.hidden_size

        pooled_output = self.pooling[task_num](sequence_output)
        pooled_output = self.dropout(self.act(pooled_output))

        logits = self.task_classifiers[task_num](pooled_output)

        outputs = (logits,)
        if labels is not None:
            assert sequence_output.shape[0] == labels.shape[0]
            loss_fct = nn.CrossEntropyLoss()
            loss = lam * loss_fct(logits.view(-1, self.task_num_labels[task_num]), labels.view(-1))
            outputs = (loss,) + outputs

        return outputs
