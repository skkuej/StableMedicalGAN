#!/usr/bin/env python
# coding: utf-8

# In[2]:


def load_data(path,size,mode):
    
    import tensorflow as tf
    feature_description = {
        't1': tf.io.FixedLenFeature([size, size], tf.float32),
        't1ce': tf.io.FixedLenFeature([size, size], tf.float32)
        }
    def augment(data):
        tf.random.set_seed(1234)
        t1, t1ce = to_3d(data)
        stacked_image = tf.stack([t1,t1ce], axis=0)
        if tf.random.uniform(())>0.5:
            stacked_image = tf.image.flip_left_right(stacked_image) 
        if tf.random.uniform(())>0.75:
            stacked_image = tf.image.rot90(stacked_image)
        t1 = stacked_image[0,:,:]
        t1ce = stacked_image[1,:,:]
        return t1, t1ce
    
    def _parse_example(example):
      # Parse the input `tf.Example` proto using the dictionary above.
      return tf.io.parse_single_example(example, feature_description)


    def to_3d(data):
        return tf.expand_dims(data['t1'],2), tf.expand_dims(data['t1ce'],2)


    if mode == 'training':
        dataset_test = tf.data.TFRecordDataset(path)
        train_ds = (
            dataset_test
            .map(_parse_example)
            .map(augment,num_parallel_calls=tf.data.experimental.AUTOTUNE)
            .batch(32)
        )
        return valid_ds
        
    elif mode == 'valid':
        dataset_test = tf.data.TFRecordDataset(path)
        valid_ds = (
            dataset_test
            .map(_parse_example)
            .map(to_3d,num_parallel_calls=tf.data.experimental.AUTOTUNE)
            .batch(32)
        )
        return valid_ds
    
    elif mode == 'test':
        dataset_test = tf.data.TFRecordDataset(path)
        test_ds = (
            dataset_test
            .map(_parse_example)
            .map(to_3d,num_parallel_calls=tf.data.experimental.AUTOTUNE)
            .batch(1)
        )
        return test_ds


# In[ ]:




