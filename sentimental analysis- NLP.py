#!/usr/bin/env python
# coding: utf-8

# In[4]:


# Install the pytorch, transformers
# pip install torch 
# pip install transformers


# In[5]:


# Import the libraries
import torch
from transformers import BertForSequenceClassification , BertTokenizer


# In[6]:


# import the data 
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased')


# In[7]:


sentences = ["I love this movie"," This is a great product", "I don't like this song"]
labels = [1,  1,  0]  # 1 for positive , 0 for negetive


# In[19]:


# tokenize the input data to convert into tenser
inputs=tokenizer(sentences,padding=True,truncation=True,return_tensors='pt')
labels=torch.tensor(labels)


# In[20]:


# optimization
optimize = torch.optim.Adam(model.parameters(),lr=0.0005,weight_decay=0.01)


# In[21]:


# train the loop
for epoch in range(3):
    model.train()
    optimize.zero_grad()
    outputs=model(**inputs,labels=labels)
    loss = outputs.loss
    loss.backward()
    optimize.step()
    print(f"Epoch {epoch+1}, Batch Loss: {loss.item()}")


# In[22]:


new_sentence = "i like song"
inputs = tokenizer(new_sentence, padding=True, truncation=True, return_tensors="pt")
with torch.no_grad():
    outputs = model(**inputs)
    predictions = torch.argmax(outputs.logits, dim=1)
    print("Sentiment:", "Positive" if predictions[0] == 1  else "Negative")


# In[23]:


new_sentence = "I don't like film"
inputs = tokenizer(new_sentence, padding=True, truncation=True, return_tensors="pt")
with torch.no_grad():
    outputs = model(**inputs)
    predictions = torch.argmax(outputs.logits, dim=1)
    print("Sentiment:", "Positive" if predictions[0] == 1  else "Negative")

