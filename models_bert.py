import torch
from torch import nn
from transformers import BertForSequenceClassification
from tqdm import tqdm

class outputclass(nn.Module): # nn.Module required or gradients won't flow (I think, it didn't work without it anyways)
    '''
    Used so that loss and logits can be called as class attributes of the 
    output of our bert model+classifier.
    This is so we can use the same train function for both BERTforsequenceclassification 
    and BERTforsequenceclassification with our custom classifier

    Example use : 
    outputs = BertWithCustomClassifier(nn.Module)
    outputs.loss
    >>> tensor
    ''' 

    def __init__(self, loss=1, logits=2):
        super().__init__()
        self.loss = loss
        self.logits = logits

class BertWithCustomClassifier(nn.Module):
    """
    Pytorch nn.Module composed of a BERT model which has it's classifier layers replaced
    with our own classifier. The classifier layers are initialised in the same 
    way as the bert layers. Using outputclass, this model outputs the same 
    .logits and .loss attributes as the unmodified BERTforsequenceclassification model.
    
    Instantiation:
        nb_hidden (int) : number of hidden units in the hidden layer of the classification head
        
    Attributes:
        forward : pytorch forward pass as usually done in a nn.Module
            tokens (tensor) : batch of input tokens
            token_type_ids (None) : only used to keep inputs consistent between models 
                                    (so that train function would work with unmodified 
                                     BERTforsequenceclassification)
            attention_mask (tensor) : batch of attention masks
            labels (tensor) : batch of labels
        
        Outputs
        freeze_bert : freezes all parameters that are not in the classification head
            freeze (bool) : True = freez the parameters ; False = don't
        
    
    """
    def __init__(self, nb_hidden=500):
        super().__init__()
        self.bert = BertForSequenceClassification.from_pretrained("bert-base-uncased",
                                                                  num_labels = 2, output_attentions = False,
                                                                  output_hidden_states = False)
  
        # Using the same dropout probability as BERTforsequenceclassification
        # https://forums.pytorchlightning.ai/t/difference-between-bertforsequenceclassification-and-bert-nn-linear/470/2
        dropout_p = self.bert.config.hidden_dropout_prob 
        
        # Replacing the default classifier layer with our own: 
        # https://github.com/huggingface/transformers/issues/1001
        # inspired from roberta's classifier https://github.com/huggingface/transformers/blob/19e5ed736611227b004c6f55679ce3536db3c28d/src/transformers/models/roberta/modeling_roberta.py#L1443
        self.bert.classifier = nn.Sequential( 
          nn.Dropout(p=dropout_p),
          nn.Linear(768,nb_hidden),
          nn.Tanh(), 
          nn.Dropout(p=dropout_p), 
          nn.Linear(nb_hidden,2),
        )
        
        # Initialisze classifier weights same way that bert does it
        # from : https://forums.pytorchlightning.ai/t/difference-between-bertforsequenceclassification-and-bert-nn-linear/470/2
        for layer in self.bert.classifier:
            self.bert._init_weights(layer)

        self.loss = nn.CrossEntropyLoss()
        self.outputclass = outputclass(loss=None,logits=None)

    def forward(self, tokens, token_type_ids, attention_mask, labels):
        
        self.outputclass.logits = self.bert(input_ids=tokens, attention_mask=attention_mask).logits
        self.outputclass.loss = self.loss(self.outputclass.logits, labels)
        
        # calling our outputclass to format the output the same way as BERTforsequenceclassification
        return self.outputclass 
        
    def freeze_bert(self, freeze=True): 
        """
        Only freez the bert layers so classifier can be trained
        Inputs
            freeze(bool) : True = freeze the parameters; False = don't
        
        outputs:
        None
        """
        for param in self.bert.bert.parameters(): # from: https://github.com/huggingface/transformers/issues/400
            param.requires_grad = (not freeze)