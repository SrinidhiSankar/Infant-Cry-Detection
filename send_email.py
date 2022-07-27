#!/usr/bin/env python
# coding: utf-8

# In[3]:


import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

def send_email(reason):
    
    email = "tosrinidhi11@gmail.com"
    fromaddr = "srpprojectns@yahoo.com" # insert sender email address here
    toaddr = email
    msg = MIMEMultipart()
    msg['From'] = fromaddr
    msg['To'] = toaddr
    msg['Subject'] = "Notification from Baby Ready"
     
    body = "Your child is currently crying likely due to " + reason
    msg.attach(MIMEText(body, 'plain'))
     
    server = smtplib.SMTP_SSL('smtp.mail.yahoo.com', 465)
    #server.starttls()
    server.login(fromaddr, "xgpnngpcqmvvhfxs") # insert sender email password here
    text = msg.as_string()
    server.sendmail(fromaddr, toaddr, text)
    server.quit()


# In[3]:


if __name__ == "__main__":
    send_email("Hunger")
    print()


# In[ ]:




