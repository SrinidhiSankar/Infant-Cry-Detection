#!/usr/bin/env python
# coding: utf-8

# In[1]:


pip install keyboard


# In[2]:


pip install os-sys


# In[3]:


import keyboard
import os
from pygame import mixer


# In[4]:


def play_audio(filename):
    
    print(os.path.dirname(os.path.abspath('__file__')) + "\\" + filename)

    mixer.init()
    mixer.music.load(os.path.dirname(os.path.abspath('__file__')) + "\\" + filename)
    mixer.music.play()
    while mixer.music.get_busy() == True:
        try: 
            if keyboard.is_pressed('esc'):
                print('Music Stopped')
                mixer.music.stop()
                break
            else:
                pass
        except:
            continue


# In[6]:


if __name__ == "__main__":
    play_audio("lullaby_goodnight.mp3")


# In[ ]:




