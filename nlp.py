#!/usr/bin/env python
# coding: utf-8

# In[1]:


pip install google-api-core


# In[2]:


pip install google-cloud-dialogflow


# In[3]:


pip install google-cloud-dialogflow-cx


# In[4]:


from google.cloud import dialogflowcx_v3beta1 as dialogflow


# In[5]:


pip install dialogflow


# In[6]:


pip install dialogflow-fulfillment


# In[13]:


import dialogflow

def detect_intent_texts(project_id, session_id, texts, language_code):
    """Returns the result of detect intent with texts as inputs.
    Using the same `session_id` between requests allows continuation
    of the conversaion."""
    session_client = dialogflow.SessionsClient()

    session = session_client.session_path(project_id, session_id)
    for text in texts:
        text_input = dialogflow.types.TextInput(
            text=text, language_code=language_code)

        query_input = dialogflow.types.QueryInput(text=text_input)

        response = session_client.detect_intent(
            session=session, query_input=query_input)
        print('You: "{}"'.format(response.query_result.query_text))
        print('I think you said: "{}"'.format(
            response.query_result.intent.display_name))
    return str(response.query_result.fulfillment_text)

if __name__ == "__main__":
    detect_intent_texts("infantcrydetect","1",["burp my baby",], "en") #insert google cloud project id


# In[8]:


def implicit():
    from google.cloud import storage

    # If you don't specify credentials when constructing the client, the
    # client library will look for credentials in the environment.
    storage_client = storage.Client()

    # Make an authenticated API request
    buckets = list(storage_client.list_buckets())
    print(buckets)


# In[12]:


#import dialogflow

def detect_intent_texts(project_id, texts, language_code):
    """Returns the result of detect intent with texts as inputs.
    Using the same `session_id` between requests allows continuation
    of the conversaion."""
    #session_client = dialogflow.SessionsClient()

    #session = session_client.session_path(project_id, session_id)
    for text in texts:
        text_input = dialogflow.types.TextInput(
            text=text, language_code=language_code)

        query_input = dialogflow.types.QueryInput(text=text_input)

        response = session_client.detect_intent(
            session=session, query_input=query_input)
        print('You: "{}"'.format(response.query_result.query_text))
        print('I think you said: "{}"'.format(
            response.query_result.intent.display_name))
    return str(response.query_result.fulfillment_text)

if __name__ == "__main__":
    detect_intent_texts("infantcrydetect",["burp my baby"], "en") #insert google cloud project id


# In[ ]:




