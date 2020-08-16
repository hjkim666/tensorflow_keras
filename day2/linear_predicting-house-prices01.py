
# coding: utf-8

# ## Boston housing - linear regression

# In[1]:


import keras
keras.__version__


# In[2]:


from keras.datasets import boston_housing

(train_data, train_targets), (test_data, test_targets) = boston_housing.load_data()


# In[3]:


train_data.shape


# In[4]:


test_data.shape


# In[5]:


train_targets


# # Data preparation

# In[6]:


mean = train_data.mean(axis=0)
train_data -= mean
std = train_data.std(axis=0)
train_data /= std 

test_data -=mean 
test_data /=std


# # Modeling

# In[7]:


from keras import models
from keras import layers

def build_model():
    model = models.Sequential()
    model.add(layers.Dense(64, activation='relu'
                           , input_shape=(train_data.shape[1],)))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(1))
    model.compile(optimizer='rmsprop', loss='mse',metrics=['mae'])
    return model


# # Kfold cross validation

# In[8]:


import numpy as np

k=4
num_val_samples = len(train_data) //k
num_epochs = 100
all_scores = []

for i in range(k):
    print('처리중인 폴드 #', i)
    #검증 데이터 준비: k번째 분할 
    val_data = train_data[i * num_val_samples: (i + 1) * num_val_samples]
    val_targets = train_targets[i * num_val_samples: (i + 1)* num_val_samples]
    
    #훈련 데이터 준비: 다른 분할 전체 
    partial_train_data = np.concatenate(
        [train_data[:i * num_val_samples],
         train_data[(i + 1) * num_val_samples:]],
        axis=0)
    partial_train_targets = np.concatenate(
        [train_targets[:i * num_val_samples:], 
         train_targets[(i + 1) * num_val_samples:]],
        axis=0)
    
    model = build_model()
    model.fit(partial_train_data, partial_train_targets,
             epochs = num_epochs, batch_size=1, verbose=0)
    val_mse, val_mae = model.evaluate(val_data, val_targets, verbose=0)
    all_scores.append(val_mae)


# In[9]:


all_scores


# In[10]:


np.mean(all_scores)


# # Epoch을 늘려서 Train

# In[11]:


from keras import backend as K

K.clear_session()


# In[ ]:


num_epochs = 500
all_mae_histories = []

for i in range(k):
    print('처리중인 폴드 #', i)
    #검증 데이터 준비: k번째 분할 
    val_data = train_data[i * num_val_samples: (i + 1) * num_val_samples]
    val_targets = train_targets[i * num_val_samples: (i + 1)* num_val_samples]
    
    #훈련 데이터 준비: 다른 분할 전체 
    partial_train_data = np.concatenate(
        [train_data[:i * num_val_samples],
         train_data[(i + 1) * num_val_samples:]],
        axis=0)
    partial_train_targets = np.concatenate(
        [train_targets[:i * num_val_samples:], 
         train_targets[(i + 1) * num_val_samples:]],
        axis=0)
    
    model = build_model()
    history = model.fit(partial_train_data, partial_train_targets,
                       validation_data = (val_data, val_targets),
                       epochs=num_epochs, batch_size=1, verbose=0)
    mae_history = history.history['val_mean_absolute_error']
    all_mae_histories.append(mae_history)


# In[ ]:


average_mae_history =  [
    np.mean([x[i] for x in all_mae_historis]) for i in range(num_epochs)]


# # 시각화

# In[ ]:


import matplotlib.pyplot as plt


# In[ ]:


plt.plot(range(1, len(average_mae_history) + 1), average_mae_history)
plt.xlabel('Epochs')
plt.ylabel('Validation MAE')
plt.show()


# In[20]:


def smooth_curve(points, factor = 0.9):
    smoothed_points = []
    for point in points:
        if smoothed_points:
            previous = smoothed_points[-1]
            smoothed_points.append(previous * factor + points * (1 - factor))
        else:
            smoothed_points.append(point)
    return smoothed_points       


# In[ ]:


model = build_model() 
model.fit(train_data, train_targets, 
         epochs=80, batch_size=16, verbose=0)
test_mse_score, test_mae_score=model.evaluate(test_data, test_targets)

