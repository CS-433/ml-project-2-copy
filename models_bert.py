import torch
from torch import nn
from transformers import BertForSequenceClassification
from tqdm import tqdm

class outputclass(nn.Module): # nn.Module required or gradients won't flow (I think, it didn't work without it anyways)
    '''
    used so that loss and logits can be called as class attributes of the 
    output of our bert model+classifier

    Example use : 
    outputs = BertWithCustomClassifier(nn.Module)
    outputs.loss
    >>> tensor
    ''' 
    # why bother with this class ? 
    # So we can use the same train function as bert model we didn't modify
    def __init__(self, loss=1, logits=2):
        super().__init__()
        self.loss = loss
        self.logits = logits

class BertWithCustomClassifier(nn.Module):
    """
    Model composed of a BERT model which has it's classifier layers replaced
    with our own classifier. The classifier layers are initialised in the same 
    way as the bert layers. Using outputclass, this model outputs the same 
    .logits and .loss attributes as the usual BERT model. 
    """
    def __init__(self, nb_hidden=500):
        super().__init__()
        self.bert = BertForSequenceClassification.from_pretrained("bert-base-uncased",
                                                                  num_labels = 2, output_attentions = False,
                                                                  output_hidden_states = False)
  
        # BertForSequenceClassification.from_pretrained('bert-base-cased')
        # https://github.com/huggingface/transformers/issues/1001
        # https://forums.pytorchlightning.ai/t/difference-between-bertforsequenceclassification-and-bert-nn-linear/470/2
        dropout_p = self.bert.config.hidden_dropout_prob 
        
        # easier to initalise the weights of linear if we use sequential (idk how otherwise)
        self.bert.classifier = nn.Sequential( 
          nn.Dropout(p=dropout_p),
          nn.Linear(768,nb_hidden),
          nn.Tanh(), # inspired from roberta
          # nn.ReLU(),
          nn.Dropout(p=dropout_p), 
          nn.Linear(nb_hidden,2),
          # nn.ReLU(), # using more layers made it take longer to train so removed
          # nn.Dropout(p=dropout_p),
          # nn.Linear(25,2)
          # nn.Sigmoid()
        )
        
        # Init classifier weights same way that bert does it
        # from : https://forums.pytorchlightning.ai/t/difference-between-bertforsequenceclassification-and-bert-nn-linear/470/2
        for layer in self.bert.classifier:
            self.bert._init_weights(layer)

        self.loss = nn.CrossEntropyLoss()
        self.outputclass = outputclass(loss=None,logits=None)

    def forward(self, tokens, token_type_ids, attention_mask, labels):
        
        # bert_output = self.bert(input_ids=tokens, attention_mask=attention_mask)
        # loss = self.loss(bert_output.logits, labels)

        self.outputclass.logits = self.bert(input_ids=tokens, attention_mask=attention_mask).logits
        self.outputclass.loss = self.loss(self.outputclass.logits, labels)

        return self.outputclass # outputclass(loss=loss,logits=bert_output.logits) # loss, bert_output.logits
        
    def freeze_bert(self, freeze=True): # 
        """
        Only freez the bert layers so classifier can be trained # our comment
        """
        for param in self.bert.bert.parameters(): # from:  https://github.com/huggingface/transformers/issues/400
            param.requires_grad = (not freeze)