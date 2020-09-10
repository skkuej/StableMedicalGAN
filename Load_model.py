#!/usr/bin/env python
# coding: utf-8

# In[1]:



def tumor_resunet():
    
    from tensorflow.keras.backend import int_shape
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import Conv2D, Conv3D, MaxPooling2D, MaxPooling3D, UpSampling2D, UpSampling3D, Add, BatchNormalization, Input, Activation, Lambda, Concatenate
    import tensorflow as tf
    from tensorflow import keras

    import os
    import time
    import datetime
    from matplotlib import pyplot as plt
    from IPython import display
    import numpy as np
    import pandas as pd

    def res_unet(filter_root, depth, input_size=(128, 128, 1), activation='relu', batch_norm=True, final_activation='tanh'):

        inputs = Input(input_size)
        x = inputs
        # Dictionary for long connections
        long_connection_store = {}

        Conv = Conv2D
        MaxPooling = MaxPooling2D
        UpSampling = UpSampling2D


        # Down sampling
        for i in range(depth):
            out_channel = 2**i * filter_root

            res = Conv(out_channel, kernel_size=1, padding='same', use_bias=False, name="Identity{}_1".format(i))(x)
            conv1 = Conv(out_channel, kernel_size=3, padding='same', name="dConv{}_1".format(i))(x)
            if batch_norm:
                conv1 = BatchNormalization(name="BN{}_1".format(i))(conv1)
            act1 = Activation(activation, name="Act{}_1".format(i))(conv1)
            conv2 = Conv(out_channel, kernel_size=3, padding='same', name="ddConv{}_2".format(i))(act1)
            if batch_norm:
                conv2 = BatchNormalization(name="BN{}_2".format(i))(conv2)
            resconnection = Add(name="Add{}_1".format(i))([res, conv2])
            act2 = Activation('selu', name="Act{}_2".format(i))(resconnection)

            if i < depth - 1:
                long_connection_store[str(i)] = act2
                x = downsample(out_channel,3)(act2)
            else:
                x = act2

        # Upsampling
        for i in range(depth - 2, -1, -1):
            out_channel = 2**(i) * filter_root
            long_connection = long_connection_store[str(i)]

            # up1 = UpSampling(name="UpSampling{}_1".format(i))(x)
            up1 = upsample(out_channel,3)(x)
            up_conv1 = Conv(out_channel, 2, activation='relu', padding='same', name="upConv{}_1".format(i))(up1)
            up_conc = Concatenate(axis=-1, name="upConcatenate{}_1".format(i))([up_conv1, long_connection])

            up_conv2 = Conv(out_channel, 3, padding='same', name="uupConv{}_1".format(i))(up_conc)
            if batch_norm:
                up_conv2 = BatchNormalization(name="upBN{}_1".format(i))(up_conv2)
            up_act1 = Activation(activation, name="upAct{}_1".format(i))(up_conv2)
            up_conv2 = Conv(out_channel, 3, padding='same', name="upConv{}_2".format(i))(up_act1)
            if batch_norm:
                up_conv2 = BatchNormalization(name="upBN{}_2".format(i))(up_conv2)

            res = Conv(out_channel, kernel_size=1, padding='same', use_bias=False, name="upIdentity{}_1".format(i))(up_conc)
            resconnection = Add(name="upAdd{}_1".format(i))([res, up_conv2])
            x = Activation(activation, name="upAct{}_2".format(i))(resconnection)

            if i == 2: x = attention(x, x.shape[3], size=1)
        # Final convolution
        output = Conv(1, 1, padding='same', activation=final_activation, name='output')(x)

        return Model(inputs, outputs=output, name='Res-UNet')

    OUTPUT_CHANNELS = 1
    def downsample(filters, size, apply_batchnorm=True):
        initializer = tf.keras.initializers.he_normal()
        result = tf.keras.Sequential()
        result.add(
          SpectralNormalization(tf.keras.layers.Conv2D(filters, size, strides=2, padding='same',
                                 kernel_initializer=initializer, use_bias=False)))

        if apply_batchnorm: 
            result.add(tf.keras.layers.BatchNormalization())

        result.add(tf.keras.layers.LeakyReLU())

        return result


    def upsample(filters, size, apply_dropout=False):
        initializer = tf.keras.initializers.he_normal()
        result = tf.keras.Sequential()
        result.add(
            SpectralNormalization(tf.keras.layers.Conv2DTranspose(filters, size, strides=2,
                                        padding='same', kernel_initializer=initializer, use_bias=False)))

        result.add(tf.keras.layers.BatchNormalization())
        if apply_dropout: result.add(tf.keras.layers.Dropout(0.5))

        result.add(tf.keras.layers.ReLU())
        return result
    class SpectralNormalization(tf.keras.layers.Wrapper):
        def __init__(self, layer, iteration=1, eps=1e-12, training=True, **kwargs):
            self.iteration = iteration
            self.eps = eps
            self.do_power_iteration = training
            if not isinstance(layer, tf.keras.layers.Layer):
                raise ValueError(
                    'Please initialize `TimeDistributed` layer with a '
                    '`Layer` instance. You passed: {input}'.format(input=layer))
            super(SpectralNormalization, self).__init__(layer, **kwargs)

        def build(self, input_shape):
            self.layer.build(input_shape)

            self.w = self.layer.kernel
            self.w_shape = self.w.shape.as_list()

            self.v = self.add_weight(shape=(1, self.w_shape[0] * self.w_shape[1] * self.w_shape[2]),
                                     initializer=tf.initializers.TruncatedNormal(stddev=0.02),
                                     trainable=False,
                                     name='sn_v',
                                     dtype=tf.float32)

            self.u = self.add_weight(shape=(1, self.w_shape[-1]),
                                     initializer=tf.initializers.TruncatedNormal(stddev=0.02),
                                     trainable=False,
                                     name='sn_u',
                                     dtype=tf.float32)

            super(SpectralNormalization, self).build()

        def call(self, inputs):
            self.update_weights()
            output = self.layer(inputs)
            self.restore_weights()  # Restore weights because of this formula "W = W - alpha * W_SN`"
            return output

        def update_weights(self):
            w_reshaped = tf.reshape(self.w, [-1, self.w_shape[-1]])

            u_hat = self.u
            v_hat = self.v  # init v vector

            if self.do_power_iteration:
                for _ in range(self.iteration):
                    v_ = tf.matmul(u_hat, tf.transpose(w_reshaped))
                    v_hat = v_ / (tf.reduce_sum(v_**2)**0.5 + self.eps)

                    u_ = tf.matmul(v_hat, w_reshaped)
                    u_hat = u_ / (tf.reduce_sum(u_**2)**0.5 + self.eps)

            sigma = tf.matmul(tf.matmul(v_hat, w_reshaped), tf.transpose(u_hat))
            self.u.assign(u_hat)
            self.v.assign(v_hat)

            self.layer.kernel.assign(self.w / sigma)

        def restore_weights(self):
            self.layer.kernel.assign(self.w)


    def hw_flatten(i) :
        sh_list = tf.shape(i)
        return tf.reshape(i, shape=[sh_list[0],sh_list[1]*sh_list[2],sh_list[3]])

    def attention(x, channels, size):
        initializer = tf.initializers.GlorotNormal()
        f = tf.keras.layers.Conv2D(channels // 8, size, strides=1, padding='same', 
                                 kernel_initializer=initializer, use_bias=False)(x) # [bs, h, w, c//8] 8
        g = tf.keras.layers.Conv2D(channels // 8, size, strides=1, padding='same', 
                                 kernel_initializer=initializer, use_bias=False)(x) # [bs, h, w, c//8] 8
        h = tf.keras.layers.Conv2D(channels, size, strides=1, padding='same', 
                                 kernel_initializer=initializer, use_bias=False)(x) # [bs, h, w, c//2] 32

        # N = h * w
        s = tf.matmul(hw_flatten(g), hw_flatten(f), transpose_b=True) # [bs, N, N]
        beta = tf.nn.softmax(s)  # attention map
        o = tf.matmul(beta, hw_flatten(h)) # [bs, N, C]
        gamma = tf.Variable(initial_value=[1.])#, initial_value=tf.constant(0.0))

        o = tf.reshape(o, shape = tf.shape(x)) # [bs, h, w, C]
        o = tf.keras.layers.Conv2D(channels, kernel_size=1, strides=1)(o)
        x = gamma * o + x
        return x
    def Discriminator_RES():
        initializer = tf.initializers.GlorotNormal()
        inp = tf.keras.layers.Input(shape=[128,128,1], name='input_image', dtype="float32")
        tar = tf.keras.layers.Input(shape=[128,128,1], name='target_image', dtype="float32")
        x = tf.keras.layers.concatenate([inp, tar]) 

        down1 = downsample(64, 4, False)(x) # (bs, 256, 256, 2) -> (bs, 128, 128, 64)
        down2 = downsample(128, 4)(down1)  # (bs, 128, 128, 64) -> (bs, 64,64, 128) 
        # att1 = attention(down3, down3.shape[3], size=1)

        zero_pad1 = tf.keras.layers.ZeroPadding2D()(down2) # (bs, 34, 34, 256)
        conv = tf.keras.layers.Conv2D(256, 4, strides=1,
                                    kernel_initializer=initializer,use_bias=False)(zero_pad1) # (bs, 34-4+1=31, 31, 256)

        batchnorm1 = tf.keras.layers.BatchNormalization()(conv)
        leaky_relu = tf.keras.layers.LeakyReLU()(batchnorm1)
        att2 = attention(leaky_relu, leaky_relu.shape[3], size=1)

        zero_pad2 = tf.keras.layers.ZeroPadding2D()(att2) # (bs, 33, 33, 256)
        last = tf.keras.layers.Conv2D(1, 4, strides=1,
                                    kernel_initializer=initializer)(zero_pad2) # (bs, 33-4+1=30, 30, 1)

        return tf.keras.Model(inputs=[inp, tar], outputs=last)

    LAMBDA = 100
    loss_object = tf.keras.losses.Hinge()
    generator_optimizer = tf.keras.optimizers.Adam(1e-4, beta_1=0.5)
    discriminator_optimizer = tf.keras.optimizers.Adam(4e-4, beta_1=0.5)

    def generator_loss(disc_generated_output, gen_output, target):

        roi_tar = tf.image.central_crop(target, central_fraction=0.3)
        roi_gen = tf.image.central_crop(gen_output,central_fraction=0.3)
        roi_loss = (LAMBDA*1.2) *tf.reduce_mean(tf.abs(roi_tar-roi_gen))

        gan_loss = loss_object(tf.ones_like(disc_generated_output), disc_generated_output)
        l1_loss = LAMBDA * tf.reduce_mean(tf.abs(target - gen_output))
        total_gen_loss = gan_loss + l1_loss + roi_loss
        return total_gen_loss, gan_loss, l1_loss, roi_loss

    def discriminator_loss(disc_real_output, disc_generated_output):
        real_loss = loss_object(tf.ones_like(disc_real_output), disc_real_output)
        generated_loss = loss_object(tf.zeros_like(disc_generated_output), disc_generated_output)
        total_disc_loss = real_loss + generated_loss
        return total_disc_loss, generated_loss, real_loss

    def ssim_function(x,y):
        C1 = np.square(0.01*2)
        C2 = np.square(0.03*2)

        mean_x = np.mean(x)
        mean_y = np.mean(y)
        std_x = np.std(x)
        std_y = np.std(y)
        cov_xy = np.cov((y.numpy().flatten(),x.numpy().flatten())) # covariance 2x2 matrix
        numerator = (2*mean_x*mean_y +C1 )*(2*np.mean(cov_xy) + C2)
        denominator = (np.square(mean_x)+np.square(mean_y)+C1)*(np.square(std_x)+np.square(std_y)+C2)
        return numerator/denominator


    res_generator = res_unet(64,4)
    res_generator.load_weights(r'F:\gan\model\ori\breast_res_model\g471_20200820-054106')
    res_discriminator = Discriminator_RES()
    res_discriminator.load_weights(r'F:\gan\model\ori\breast_res_model\d471_20200820-054110')
    return res_generator, res_discriminator


# In[2]:




def tumor_sagan():
    import tensorflow as tf
    from tensorflow import keras

    import os
    import time
    import datetime
    from matplotlib import pyplot as plt
    import numpy as np
    import pandas as pd
    # SAGAN - BOX loss
    class SpectralNormalization(tf.keras.layers.Wrapper):
        def __init__(self, layer, iteration=1, eps=1e-12, training=True, **kwargs):
            self.iteration = iteration
            self.eps = eps
            self.do_power_iteration = training
            if not isinstance(layer, tf.keras.layers.Layer):
                raise ValueError(
                    'Please initialize `TimeDistributed` layer with a '
                    '`Layer` instance. You passed: {input}'.format(input=layer))
            super(SpectralNormalization, self).__init__(layer, **kwargs)

        def build(self, input_shape):
            self.layer.build(input_shape)

            self.w = self.layer.kernel
            self.w_shape = self.w.shape.as_list()

            self.v = self.add_weight(shape=(1, self.w_shape[0] * self.w_shape[1] * self.w_shape[2]),
                                     initializer=tf.initializers.TruncatedNormal(stddev=0.02),
                                     trainable=False,
                                     name='sn_v',
                                     dtype=tf.float32)

            self.u = self.add_weight(shape=(1, self.w_shape[-1]),
                                     initializer=tf.initializers.TruncatedNormal(stddev=0.02),
                                     trainable=False,
                                     name='sn_u',
                                     dtype=tf.float32)

            super(SpectralNormalization, self).build()

        def call(self, inputs):
            self.update_weights()
            output = self.layer(inputs)
            self.restore_weights()  # Restore weights because of this formula "W = W - alpha * W_SN`"
            return output

        def update_weights(self):
            w_reshaped = tf.reshape(self.w, [-1, self.w_shape[-1]])

            u_hat = self.u
            v_hat = self.v  # init v vector

            if self.do_power_iteration:
                for _ in range(self.iteration):
                    v_ = tf.matmul(u_hat, tf.transpose(w_reshaped))
                    v_hat = v_ / (tf.reduce_sum(v_**2)**0.5 + self.eps)

                    u_ = tf.matmul(v_hat, w_reshaped)
                    u_hat = u_ / (tf.reduce_sum(u_**2)**0.5 + self.eps)

            sigma = tf.matmul(tf.matmul(v_hat, w_reshaped), tf.transpose(u_hat))
            self.u.assign(u_hat)
            self.v.assign(v_hat)

            self.layer.kernel.assign(self.w / sigma)

        def restore_weights(self):
            self.layer.kernel.assign(self.w)

    OUTPUT_CHANNELS = 1
    def downsample(filters, size, apply_batchnorm=True, apply_residual=True):
        initializer = tf.keras.initializers.he_normal()
        result = tf.keras.Sequential()
        result.add(
          SpectralNormalization(tf.keras.layers.Conv2D(filters, size, strides=2, padding='same',
                                 kernel_initializer=initializer, use_bias=False)))

        if apply_batchnorm:
            result.add(tf.keras.layers.BatchNormalization())
        result.add(tf.keras.layers.LeakyReLU())

        return result


    def upsample(filters, size, apply_dropout=False):
        initializer = tf.keras.initializers.he_normal()
        result = tf.keras.Sequential()
        result.add(
            SpectralNormalization(tf.keras.layers.Conv2DTranspose(filters, size, strides=2,
                                        padding='same', kernel_initializer=initializer, use_bias=False)))

        result.add(tf.keras.layers.BatchNormalization())
        if apply_dropout:
            result.add(tf.keras.layers.Dropout(0.5))
        result.add(tf.keras.layers.ReLU())
        return result

    def hw_flatten(i) :
        sh_list = tf.shape(i)
        return tf.reshape(i, shape=[sh_list[0],sh_list[1]*sh_list[2],sh_list[3]])

    def attention(x, channels, size):
        initializer = tf.initializers.GlorotNormal()
        f = tf.keras.layers.Conv2D(channels // 8, size, strides=1, padding='same', 
                                 kernel_initializer=initializer, use_bias=False)(x) # [bs, h, w, c//8] 8
        g = tf.keras.layers.Conv2D(channels // 8, size, strides=1, padding='same', 
                                 kernel_initializer=initializer, use_bias=False)(x) # [bs, h, w, c//8] 8
        h = tf.keras.layers.Conv2D(channels, size, strides=1, padding='same', 
                                 kernel_initializer=initializer, use_bias=False)(x) # [bs, h, w, c//2] 32

        # N = h * w
        s = tf.matmul(hw_flatten(g), hw_flatten(f), transpose_b=True) # [bs, N, N]
        beta = tf.nn.softmax(s)  # attention map
        o = tf.matmul(beta, hw_flatten(h)) # [bs, N, C]
        gamma = tf.Variable(initial_value=[1.])#, initial_value=tf.constant(0.0))

        o = tf.reshape(o, shape = tf.shape(x)) # [bs, h, w, C]
        o = tf.keras.layers.Conv2D(channels, kernel_size=1, strides=1)(o)
        x = gamma * o + x
        return x
    def Generator_SA():
        inputs = tf.keras.layers.Input(shape=[128,128,1], dtype="float32")

        down_stack = [
        downsample(32, 4, apply_batchnorm=False), # (bs, 64, 64, 32)
        downsample(64, 4), # (bs, 32, 32, 64)
        downsample(128, 4), # (bs, 16, 16, 128)
        downsample(256, 4), # (bs, 8, 8, 256)
        downsample(256, 4), # (bs, 4, 4, 256)
        downsample(256, 4), # (bs, 2, 2, 256)
        downsample(256, 4), # (bs, 1, 1, 256)
        ]

        up_stack = [
        upsample(256, 4, apply_dropout=True), # (bs, 4, 4, 256)
        upsample(256, 4, apply_dropout=True), # (bs, 8, 8, 256)
        upsample(256, 4), # (bs, 16, 16, 256)
        upsample(128, 4), # (bs, 32, 32, 128)
        upsample(64, 4), # (bs, 64, 64, 64)
        # self-attention layer
        upsample(32, 4), # (bs,128, 128, 32)
        # self-attention layer
        ]

        # initializer = tf.random_normal_initializer(0., 0.02)
        initializer = tf.initializers.GlorotNormal()
        # initializer = tf.keras.initializers.he_normal()

        last = tf.keras.layers.Conv2DTranspose(OUTPUT_CHANNELS, 4,
                                             strides=2,
                                             padding='same',
                                             kernel_initializer=initializer, activation='tanh') # (bs, 256,256,1)

        x = inputs

        # Downsampling through the model
        skips = []
        for down in down_stack:
            x = down(x)
            skips.append(x)

        skips = reversed(skips[:-1])

        cnt = 1
        # Upsampling and establishing the skip connections
        for up, skip in zip(up_stack, skips):

            x = up(x)
            x = tf.keras.layers.Concatenate()([x,skip])     
            if x.shape[3] == 128 :#or x.shape[3] == 32
                x_pre = x
                x = attention(x, x.shape[3], size=1)

        x = last(x)
        x = tf.cast(x, tf.float32)
        return tf.keras.Model(inputs=inputs, outputs=x)


    def Discriminator_SA():
        initializer = tf.initializers.GlorotNormal()
        inp = tf.keras.layers.Input(shape=[128,128,1], name='input_image', dtype="float32")
        tar = tf.keras.layers.Input(shape=[128,128,1], name='target_image', dtype="float32")
        x = tf.keras.layers.concatenate([inp, tar]) 

        down1 = downsample(64, 4, False)(x) # (bs, 256, 256, 2) -> (bs, 128, 128, 64)
        down2 = downsample(128, 4)(down1)  # (bs, 128, 128, 64) -> (bs, 64,64, 128) 
        # att1 = attention(down3, down3.shape[3], size=1)

        zero_pad1 = tf.keras.layers.ZeroPadding2D()(down2) # (bs, 34, 34, 256)
        conv = tf.keras.layers.Conv2D(256, 4, strides=1,
                                    kernel_initializer=initializer,use_bias=False)(zero_pad1) # (bs, 34-4+1=31, 31, 256)

        batchnorm1 = tf.keras.layers.BatchNormalization()(conv)
        leaky_relu = tf.keras.layers.LeakyReLU()(batchnorm1)
        att2 = attention(leaky_relu, leaky_relu.shape[3], size=1)

        zero_pad2 = tf.keras.layers.ZeroPadding2D()(att2) # (bs, 33, 33, 256)
        last = tf.keras.layers.Conv2D(1, 4, strides=1,
                                    kernel_initializer=initializer)(zero_pad2) # (bs, 33-4+1=30, 30, 1)

        return tf.keras.Model(inputs=[inp, tar], outputs=last)

 

    generator_sa = Generator_SA()
    discriminator_sa = Discriminator_SA()
    generator_sa.load_weights(r'F:\gan\model\ori\breast_loss_model\g510_20200816-013729')
    discriminator_sa.load_weights(r'F:\gan\model\ori\breast_loss_model\d510_20200816-013730')
    
    return generator_sa, discriminator_sa


# In[3]:



def tumor_origin():
    import tensorflow as tf
    from tensorflow import keras

    import os
    import time
    import datetime
    from matplotlib import pyplot as plt
    import numpy as np
    import pandas as pd

    OUTPUT_CHANNELS = 1
    def downsample(filters, size, apply_batchnorm=True):
        initializer = tf.random_normal_initializer(0., 0.02)
        result = tf.keras.Sequential()
        result.add(
          tf.keras.layers.Conv2D(filters, size, strides=2, padding='same',
                                 kernel_initializer=initializer, use_bias=False))
        if apply_batchnorm:
            result.add(tf.keras.layers.BatchNormalization())
        result.add(tf.keras.layers.LeakyReLU())
        return result


    def upsample(filters, size, apply_dropout=False):
        initializer = tf.random_normal_initializer(0., 0.02)
        result = tf.keras.Sequential()
        result.add(
        tf.keras.layers.Conv2DTranspose(filters, size, strides=2,
                                        padding='same',
                                        kernel_initializer=initializer,
                                        use_bias=False))
        result.add(tf.keras.layers.BatchNormalization())
        if apply_dropout:
            result.add(tf.keras.layers.Dropout(0.5))
        result.add(tf.keras.layers.ReLU())
        return result

    def Generator():
        inputs = tf.keras.layers.Input(shape=[128,128,1], dtype="float32") 

        down_stack = [
        downsample(32, 4, apply_batchnorm=False), # (bs, 64, 64, 32)
        downsample(64, 4), # (bs, 32, 32, 64)
        downsample(128, 4), # (bs, 16, 16, 128)
        downsample(256, 4), # (bs, 8, 8, 256)
        downsample(256, 4), # (bs, 4, 4, 256)
        downsample(256, 4), # (bs, 2, 2, 256)
        downsample(256, 4) # (bs, 1, 1, 256)
        ]

        up_stack = [
        upsample(256, 4, apply_dropout=True), # (bs, 4, 4, 256)
        upsample(256, 4, apply_dropout=True), # (bs, 8, 8, 256)
        upsample(256, 4), # (bs, 16, 16, 256)
        upsample(128, 4), # (bs, 32, 32, 128)
        upsample(64, 4), # (bs, 64, 64, 64)
        upsample(32, 4)
        ]

        initializer = tf.random_normal_initializer(0., 0.02)

        last = tf.keras.layers.Conv2DTranspose(OUTPUT_CHANNELS, 4,
                                             strides=2,
                                             padding='same',
                                             kernel_initializer=initializer, activation='tanh') # (bs, 128,128,1)

        x = inputs

      # Downsampling through the model
        skips = []
        for down in down_stack:
            x = down(x)
            skips.append(x)

        skips = reversed(skips[:-1])

      # Upsampling and establishing the skip connections
        for up, skip in zip(up_stack, skips):
            x = up(x)
            x = tf.keras.layers.Concatenate()([x, skip])

        x = last(x)
        x = tf.cast(x, tf.float32)
        return tf.keras.Model(inputs=inputs, outputs=x)


    def Discriminator():
        initializer = tf.random_normal_initializer(0., 0.02)

        inp = tf.keras.layers.Input(shape=[128,128,1], name='input_image', dtype="float32")
        tar = tf.keras.layers.Input(shape=[128,128,1], name='target_image', dtype="float32")

        x = tf.keras.layers.concatenate([inp, tar]) # (bs, 128,128,channel*2=2)
        down1 = downsample(64, 4, False)(x) # (bs, 64,64, 64) batchnorm = False ?? 
        down2 = downsample(128, 4)(down1) # (bs, 32, 32, 128)

        zero_pad1 = tf.keras.layers.ZeroPadding2D()(down2) # (bs, 34, 34, 256)
        conv = tf.keras.layers.Conv2D(256, 4, strides=1,
                                    kernel_initializer=initializer,
                                    use_bias=False)(zero_pad1) # (bs, 34-4+1=31, 31, 256)

        batchnorm1 = tf.keras.layers.BatchNormalization()(conv)
        leaky_relu = tf.keras.layers.LeakyReLU()(batchnorm1)
        zero_pad2 = tf.keras.layers.ZeroPadding2D()(leaky_relu) # (bs, 33, 33, 256)
        last = tf.keras.layers.Conv2D(1, 4, strides=1,
                                    kernel_initializer=initializer)(zero_pad2) # (bs, 33-4+1=30, 30, 1)

        return tf.keras.Model(inputs=[inp, tar], outputs=last)

    generator = Generator()
    discriminator = Discriminator()
    LAMBDA = 100
    loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
    discriminator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5) # default beta 1 : 0.9 (momentum wieght)


    generator = Generator()
    generator.load_weights(r'F:\gan\model\ori\breast_ori_model\g200_20200814-154426')
    discriminator = Discriminator()
    discriminator.load_weights(r'F:\gan\model\ori\breast_ori_model\d200_20200814-154426')
    return generator, discriminator

def mv_resunet():
    import tensorflow as tf
    from tensorflow import keras

    import os
    import time
    import datetime
    from matplotlib import pyplot as plt
    import numpy as np
    import pandas as pd
    
    
    ################################# MV-RES U-net #########################################
    from tensorflow.keras.backend import int_shape
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import Conv2D, Conv3D, MaxPooling2D, MaxPooling3D, UpSampling2D, UpSampling3D, Add, BatchNormalization, Input, Activation, Lambda, Concatenate


    def res_unet(filter_root, depth, input_size=(100, 100, 1), activation='relu', batch_norm=True, final_activation='tanh'):

        inputs = Input(input_size)
        x = inputs
        # Dictionary for long connections
        long_connection_store = {}

        Conv = Conv2D
        MaxPooling = MaxPooling2D
        UpSampling = UpSampling2D


        # Down sampling
        for i in range(depth):
            out_channel = 2**i * filter_root

            res = Conv(out_channel, kernel_size=1, padding='same', use_bias=False, name="Identity{}_1".format(i))(x)
            conv1 = Conv(out_channel, kernel_size=3, padding='same', name="dConv{}_1".format(i))(x)
            if batch_norm:
                conv1 = BatchNormalization(name="BN{}_1".format(i))(conv1)
            act1 = Activation(activation, name="Act{}_1".format(i))(conv1)
            conv2 = Conv(out_channel, kernel_size=3, padding='same', name="ddConv{}_2".format(i))(act1)
            if batch_norm:
                conv2 = BatchNormalization(name="BN{}_2".format(i))(conv2)
            resconnection = Add(name="Add{}_1".format(i))([res, conv2])
            act2 = Activation('selu', name="Act{}_2".format(i))(resconnection)

            if i < depth - 1:
                long_connection_store[str(i)] = act2
                x = downsample(out_channel,3)(act2)
            else:
                x = act2

        # Upsampling
        for i in range(depth - 2, -1, -1):
            out_channel = 2**(i) * filter_root
            long_connection = long_connection_store[str(i)]

            # up1 = UpSampling(name="UpSampling{}_1".format(i))(x)
            up1 = upsample(out_channel,3)(x)
            up_conv1 = Conv(out_channel, 2, activation='relu', padding='same', name="upConv{}_1".format(i))(up1)
            if up_conv1.shape[1]==8 or up_conv1.shape[1]==14 or up_conv1.shape[1]==26:
                up_conv1 = up_conv1[:,:up_conv1.shape[1]-1,:up_conv1.shape[1]-1,:]

            up_conc = Concatenate(axis=-1, name="upConcatenate{}_1".format(i))([up_conv1, long_connection])

            up_conv2 = Conv(out_channel, 3, padding='same', name="uupConv{}_1".format(i))(up_conc)
            if batch_norm:
                up_conv2 = BatchNormalization(name="upBN{}_1".format(i))(up_conv2)
            up_act1 = Activation(activation, name="upAct{}_1".format(i))(up_conv2)
            up_conv2 = Conv(out_channel, 3, padding='same', name="upConv{}_2".format(i))(up_act1)
            if batch_norm:
                up_conv2 = BatchNormalization(name="upBN{}_2".format(i))(up_conv2)

            res = Conv(out_channel, kernel_size=1, padding='same', use_bias=False, name="upIdentity{}_1".format(i))(up_conc)
            resconnection = Add(name="upAdd{}_1".format(i))([res, up_conv2])
            x = Activation(activation, name="upAct{}_2".format(i))(resconnection)

            if i == 2: x = attention(x, x.shape[3], size=1)
        # Final convolution
        output = Conv(1, 1, padding='same', activation=final_activation, name='output')(x)

        return Model(inputs, outputs=output, name='Res-UNet')

    OUTPUT_CHANNELS = 1
    def downsample(filters, size, apply_batchnorm=True):
        initializer = tf.keras.initializers.he_normal()
        result = tf.keras.Sequential()
        result.add(
          SpectralNormalization(tf.keras.layers.Conv2D(filters, size, strides=2, padding='same',
                                 kernel_initializer=initializer, use_bias=False)))

        if apply_batchnorm: 
            result.add(tf.keras.layers.BatchNormalization())

        result.add(tf.keras.layers.LeakyReLU())

        return result


    def upsample(filters, size, apply_dropout=False):
        initializer = tf.keras.initializers.he_normal()
        result = tf.keras.Sequential()
        result.add(
            SpectralNormalization(tf.keras.layers.Conv2DTranspose(filters, size, strides=2,
                                        padding='same', kernel_initializer=initializer, use_bias=False)))

        result.add(tf.keras.layers.BatchNormalization())
        if apply_dropout: result.add(tf.keras.layers.Dropout(0.5))

        result.add(tf.keras.layers.ReLU())
        return result
    class SpectralNormalization(tf.keras.layers.Wrapper):
        def __init__(self, layer, iteration=1, eps=1e-12, training=True, **kwargs):
            self.iteration = iteration
            self.eps = eps
            self.do_power_iteration = training
            if not isinstance(layer, tf.keras.layers.Layer):
                raise ValueError(
                    'Please initialize `TimeDistributed` layer with a '
                    '`Layer` instance. You passed: {input}'.format(input=layer))
            super(SpectralNormalization, self).__init__(layer, **kwargs)

        def build(self, input_shape):
            self.layer.build(input_shape)

            self.w = self.layer.kernel
            self.w_shape = self.w.shape.as_list()

            self.v = self.add_weight(shape=(1, self.w_shape[0] * self.w_shape[1] * self.w_shape[2]),
                                     initializer=tf.initializers.TruncatedNormal(stddev=0.02),
                                     trainable=False,
                                     name='sn_v',
                                     dtype=tf.float32)

            self.u = self.add_weight(shape=(1, self.w_shape[-1]),
                                     initializer=tf.initializers.TruncatedNormal(stddev=0.02),
                                     trainable=False,
                                     name='sn_u',
                                     dtype=tf.float32)

            super(SpectralNormalization, self).build()

        def call(self, inputs):
            self.update_weights()
            output = self.layer(inputs)
            self.restore_weights()  # Restore weights because of this formula "W = W - alpha * W_SN`"
            return output

        def update_weights(self):
            w_reshaped = tf.reshape(self.w, [-1, self.w_shape[-1]])

            u_hat = self.u
            v_hat = self.v  # init v vector

            if self.do_power_iteration:
                for _ in range(self.iteration):
                    v_ = tf.matmul(u_hat, tf.transpose(w_reshaped))
                    v_hat = v_ / (tf.reduce_sum(v_**2)**0.5 + self.eps)

                    u_ = tf.matmul(v_hat, w_reshaped)
                    u_hat = u_ / (tf.reduce_sum(u_**2)**0.5 + self.eps)

            sigma = tf.matmul(tf.matmul(v_hat, w_reshaped), tf.transpose(u_hat))
            self.u.assign(u_hat)
            self.v.assign(v_hat)

            self.layer.kernel.assign(self.w / sigma)

        def restore_weights(self):
            self.layer.kernel.assign(self.w)


    def hw_flatten(i) :
        sh_list = tf.shape(i)
        return tf.reshape(i, shape=[sh_list[0],sh_list[1]*sh_list[2],sh_list[3]])

    def attention(x, channels, size):
        initializer = tf.initializers.GlorotNormal()
        f = tf.keras.layers.Conv2D(channels // 8, size, strides=1, padding='same', 
                                 kernel_initializer=initializer, use_bias=False)(x) # [bs, h, w, c//8] 8
        g = tf.keras.layers.Conv2D(channels // 8, size, strides=1, padding='same', 
                                 kernel_initializer=initializer, use_bias=False)(x) # [bs, h, w, c//8] 8
        h = tf.keras.layers.Conv2D(channels, size, strides=1, padding='same', 
                                 kernel_initializer=initializer, use_bias=False)(x) # [bs, h, w, c//2] 32

        # N = h * w
        s = tf.matmul(hw_flatten(g), hw_flatten(f), transpose_b=True) # [bs, N, N]
        beta = tf.nn.softmax(s)  # attention map
        o = tf.matmul(beta, hw_flatten(h)) # [bs, N, C]
        gamma = tf.Variable(initial_value=[1.])#, initial_value=tf.constant(0.0))

        o = tf.reshape(o, shape = tf.shape(x)) # [bs, h, w, C]
        o = tf.keras.layers.Conv2D(channels, kernel_size=1, strides=1)(o)
        x = gamma * o + x
        return x
    def Discriminator():
        initializer = tf.initializers.GlorotNormal()
        inp = tf.keras.layers.Input(shape=[100,100,1], name='input_image', dtype="float32")
        tar = tf.keras.layers.Input(shape=[100,100,1], name='target_image', dtype="float32")
        x = tf.keras.layers.concatenate([inp, tar]) 

        down1 = downsample(64, 4, False)(x) # (bs, 256, 256, 2) -> (bs, 128, 128, 64)
        down2 = downsample(128, 4)(down1)  # (bs, 128, 128, 64) -> (bs, 64,64, 128) 
        # att1 = attention(down3, down3.shape[3], size=1)

        zero_pad1 = tf.keras.layers.ZeroPadding2D()(down2) # (bs, 34, 34, 256)
        conv = tf.keras.layers.Conv2D(256, 4, strides=1,
                                    kernel_initializer=initializer,use_bias=False)(zero_pad1) # (bs, 34-4+1=31, 31, 256)

        batchnorm1 = tf.keras.layers.BatchNormalization()(conv)
        leaky_relu = tf.keras.layers.LeakyReLU()(batchnorm1)
        att2 = attention(leaky_relu, leaky_relu.shape[3], size=1)

        zero_pad2 = tf.keras.layers.ZeroPadding2D()(att2) # (bs, 33, 33, 256)
        last = tf.keras.layers.Conv2D(1, 4, strides=1,
                                    kernel_initializer=initializer)(zero_pad2) # (bs, 33-4+1=30, 30, 1)

        return tf.keras.Model(inputs=[inp, tar], outputs=last)

    discriminator = Discriminator()
    LAMBDA = 100
    loss_object = tf.keras.losses.Hinge()
    generator_optimizer = tf.keras.optimizers.Adam(1e-4, beta_1=0.5)
    discriminator_optimizer = tf.keras.optimizers.Adam(4e-4, beta_1=0.5)

    def generator_loss(disc_generated_output, gen_output, target):

        roi_tar = tf.image.central_crop(target, central_fraction=0.3)
        roi_gen = tf.image.central_crop(gen_output,central_fraction=0.3)
        roi_loss = (LAMBDA*1.2) *tf.reduce_mean(tf.abs(roi_tar-roi_gen))

        gan_loss = loss_object(tf.ones_like(disc_generated_output), disc_generated_output)
        l1_loss = LAMBDA * tf.reduce_mean(tf.abs(target - gen_output))
        total_gen_loss = gan_loss + l1_loss + roi_loss
        return total_gen_loss, gan_loss, l1_loss, roi_loss

    def discriminator_loss(disc_real_output, disc_generated_output):
        real_loss = loss_object(tf.ones_like(disc_real_output), disc_real_output)
        generated_loss = loss_object(tf.zeros_like(disc_generated_output), disc_generated_output)
        total_disc_loss = real_loss + generated_loss
        return total_disc_loss, generated_loss, real_loss
    res_generator = res_unet(64,4)
    res_discriminator = Discriminator()
    res_generator.load_weights(r'F:\gan\model\mv\breast_mv_res_model\g316_20200901-185218')
    res_discriminator.load_weights(r'F:\gan\model\mv\breast_mv_res_model\d316_20200901-185219')
    
    return res_generator, res_discriminator

def mv_sagan():
    
    import tensorflow as tf
    from tensorflow import keras

    import os
    import time
    import datetime
    from matplotlib import pyplot as plt
    import numpy as np
    import pandas as pd
    ################################### SAGAN BOX LOSS ##################################

    OUTPUT_CHANNELS = 1
    def downsample(filters, size, apply_batchnorm=True):
        initializer = tf.keras.initializers.he_normal()
        result = tf.keras.Sequential()
        result.add(
          SpectralNormalization(tf.keras.layers.Conv2D(filters, size, strides=2, padding='same',
                                 kernel_initializer=initializer, use_bias=False)))

        if apply_batchnorm: 
            result.add(tf.keras.layers.BatchNormalization())

        result.add(tf.keras.layers.LeakyReLU())

        return result


    def upsample(filters, size, apply_dropout=False):
        initializer = tf.keras.initializers.he_normal()
        result = tf.keras.Sequential()
        result.add(
            SpectralNormalization(tf.keras.layers.Conv2DTranspose(filters, size, strides=2,
                                        padding='same', kernel_initializer=initializer, use_bias=False)))

        result.add(tf.keras.layers.BatchNormalization())
        if apply_dropout: result.add(tf.keras.layers.Dropout(0.5))

        result.add(tf.keras.layers.ReLU())
        return result
    class SpectralNormalization(tf.keras.layers.Wrapper):
        def __init__(self, layer, iteration=1, eps=1e-12, training=True, **kwargs):
            self.iteration = iteration
            self.eps = eps
            self.do_power_iteration = training
            if not isinstance(layer, tf.keras.layers.Layer):
                raise ValueError(
                    'Please initialize `TimeDistributed` layer with a '
                    '`Layer` instance. You passed: {input}'.format(input=layer))
            super(SpectralNormalization, self).__init__(layer, **kwargs)

        def build(self, input_shape):
            self.layer.build(input_shape)

            self.w = self.layer.kernel
            self.w_shape = self.w.shape.as_list()

            self.v = self.add_weight(shape=(1, self.w_shape[0] * self.w_shape[1] * self.w_shape[2]),
                                     initializer=tf.initializers.TruncatedNormal(stddev=0.02),
                                     trainable=False,
                                     name='sn_v',
                                     dtype=tf.float32)

            self.u = self.add_weight(shape=(1, self.w_shape[-1]),
                                     initializer=tf.initializers.TruncatedNormal(stddev=0.02),
                                     trainable=False,
                                     name='sn_u',
                                     dtype=tf.float32)

            super(SpectralNormalization, self).build()

        def call(self, inputs):
            self.update_weights()
            output = self.layer(inputs)
            self.restore_weights()  # Restore weights because of this formula "W = W - alpha * W_SN`"
            return output

        def update_weights(self):
            w_reshaped = tf.reshape(self.w, [-1, self.w_shape[-1]])

            u_hat = self.u
            v_hat = self.v  # init v vector

            if self.do_power_iteration:
                for _ in range(self.iteration):
                    v_ = tf.matmul(u_hat, tf.transpose(w_reshaped))
                    v_hat = v_ / (tf.reduce_sum(v_**2)**0.5 + self.eps)

                    u_ = tf.matmul(v_hat, w_reshaped)
                    u_hat = u_ / (tf.reduce_sum(u_**2)**0.5 + self.eps)

            sigma = tf.matmul(tf.matmul(v_hat, w_reshaped), tf.transpose(u_hat))
            self.u.assign(u_hat)
            self.v.assign(v_hat)

            self.layer.kernel.assign(self.w / sigma)

        def restore_weights(self):
            self.layer.kernel.assign(self.w)


    def hw_flatten(i) :
        sh_list = tf.shape(i)
        return tf.reshape(i, shape=[sh_list[0],sh_list[1]*sh_list[2],sh_list[3]])

    def attention(x, channels, size):
        initializer = tf.initializers.GlorotNormal()
        f = tf.keras.layers.Conv2D(channels // 8, size, strides=1, padding='same', 
                                 kernel_initializer=initializer, use_bias=False)(x) # [bs, h, w, c//8] 8
        g = tf.keras.layers.Conv2D(channels // 8, size, strides=1, padding='same', 
                                 kernel_initializer=initializer, use_bias=False)(x) # [bs, h, w, c//8] 8
        h = tf.keras.layers.Conv2D(channels, size, strides=1, padding='same', 
                                 kernel_initializer=initializer, use_bias=False)(x) # [bs, h, w, c//2] 32

        # N = h * w
        s = tf.matmul(hw_flatten(g), hw_flatten(f), transpose_b=True) # [bs, N, N]
        beta = tf.nn.softmax(s)  # attention map
        o = tf.matmul(beta, hw_flatten(h)) # [bs, N, C]
        gamma = tf.Variable(initial_value=[1.])#, initial_value=tf.constant(0.0))

        o = tf.reshape(o, shape = tf.shape(x)) # [bs, h, w, C]
        o = tf.keras.layers.Conv2D(channels, kernel_size=1, strides=1)(o)
        x = gamma * o + x
        return x


    def Generator():
        inputs = tf.keras.layers.Input(shape=[100,100,1], dtype="float32")

        down_stack = [
        downsample(32, 4, apply_batchnorm=False), # (bs, 64, 64, 32)
        downsample(64, 4), # (bs, 32, 32, 64)
        downsample(128, 4), # (bs, 16, 16, 128)
        downsample(256, 4), # (bs, 8, 8, 256)
        downsample(256, 4), # (bs, 4, 4, 256)
        downsample(256, 4), # (bs, 2, 2, 256)
        downsample(256, 4), # (bs, 1, 1, 256)
        ]

        up_stack = [
        upsample(256, 4, apply_dropout=True), # (bs, 4, 4, 256)
        upsample(256, 4, apply_dropout=True), # (bs, 8, 8, 256)
        upsample(256, 4), # (bs, 16, 16, 256)
        upsample(128, 4), # (bs, 32, 32, 128)
        upsample(64, 4), # (bs, 64, 64, 64)
        # self-attention layer
        upsample(32, 4), # (bs,128, 128, 32)
        # self-attention layer
        ]

        # initializer = tf.random_normal_initializer(0., 0.02)
        initializer = tf.initializers.GlorotNormal()
        # initializer = tf.keras.initializers.he_normal()

        last = tf.keras.layers.Conv2DTranspose(OUTPUT_CHANNELS, 4,
                                             strides=2,
                                             padding='same',
                                             kernel_initializer=initializer, activation='tanh') # (bs, 256,256,1)

        x = inputs

        # Downsampling through the model
        skips = []
        for down in down_stack:
            x = down(x)
            skips.append(x)

        skips = reversed(skips[:-1])

        cnt = 1
        # Upsampling and establishing the skip connections
        for up, skip in zip(up_stack, skips):
            x = up(x)
            if x.shape[1]==8 or x.shape[1]==14 or x.shape[1]==26:
                x = x[:,:x.shape[1]-1,:x.shape[1]-1,:]
            x = tf.keras.layers.Concatenate()([x,skip])     

            if x.shape[3] == 128 :#or x.shape[3] == 32
                x_pre = x
                x = attention(x, x.shape[3], size=1)

        x = last(x)
        x = tf.cast(x, tf.float32)
        return tf.keras.Model(inputs=inputs, outputs=x)


    def Discriminator():
        initializer = tf.initializers.GlorotNormal()
        inp = tf.keras.layers.Input(shape=[100,100,1], name='input_image', dtype="float32")
        tar = tf.keras.layers.Input(shape=[100,100,1], name='target_image', dtype="float32")
        x = tf.keras.layers.concatenate([inp, tar]) 

        down1 = downsample(64, 4, False)(x) # (bs, 256, 256, 2) -> (bs, 128, 128, 64)
        down2 = downsample(128, 4)(down1)  # (bs, 128, 128, 64) -> (bs, 64,64, 128) 
        # att1 = attention(down3, down3.shape[3], size=1)

        zero_pad1 = tf.keras.layers.ZeroPadding2D()(down2) # (bs, 34, 34, 256)
        conv = tf.keras.layers.Conv2D(256, 4, strides=1,
                                    kernel_initializer=initializer,use_bias=False)(zero_pad1) # (bs, 34-4+1=31, 31, 256)

        batchnorm1 = tf.keras.layers.BatchNormalization()(conv)
        leaky_relu = tf.keras.layers.LeakyReLU()(batchnorm1)
        att2 = attention(leaky_relu, leaky_relu.shape[3], size=1)

        zero_pad2 = tf.keras.layers.ZeroPadding2D()(att2) # (bs, 33, 33, 256)
        last = tf.keras.layers.Conv2D(1, 4, strides=1,
                                    kernel_initializer=initializer)(zero_pad2) # (bs, 33-4+1=30, 30, 1)

        return tf.keras.Model(inputs=[inp, tar], outputs=last)

    LAMBDA = 100
    loss_object = tf.keras.losses.Hinge()
    generator_optimizer = tf.keras.optimizers.Adam(1e-4, beta_1=0.5)
    discriminator_optimizer = tf.keras.optimizers.Adam(4e-4, beta_1=0.5)

    generator_sa = Generator()
    discriminator_sa = Discriminator()

    generator_sa.load_weights(r'F:\gan\model\mv\breast_mv_loss_model\g350_20200902-142131')
    discriminator_sa.load_weights(r'F:\gan\model\mv\breast_mv_loss_model\d350_20200902-142133')
    return generator_sa, discriminator_sa

def mv_origin():
    
    import tensorflow as tf
    from tensorflow import keras

    import os
    import time
    import datetime
    from matplotlib import pyplot as plt
    import numpy as np
    import pandas as pd
    
    ####################### MV - original NET ###############################
    OUTPUT_CHANNELS = 1
    def downsample(filters, size, apply_batchnorm=True):
        initializer = tf.random_normal_initializer(0., 0.02)
        result = tf.keras.Sequential()
        result.add(
          tf.keras.layers.Conv2D(filters, size, strides=2, padding='same',
                                 kernel_initializer=initializer, use_bias=False))


        if apply_batchnorm:
            result.add(tf.keras.layers.BatchNormalization())
        result.add(tf.keras.layers.LeakyReLU())

        return result


    def upsample(filters, size, apply_dropout=False):
        initializer = tf.random_normal_initializer(0., 0.02)
        result = tf.keras.Sequential()
        result.add(
            tf.keras.layers.Conv2DTranspose(filters, size, strides=2, padding='same',
                                            kernel_initializer=initializer,
                                            use_bias=False))

        result.add(tf.keras.layers.BatchNormalization())

        if apply_dropout:
            result.add(tf.keras.layers.Dropout(0.5))
        result.add(tf.keras.layers.ReLU())
        return result


    def Generator():
        inputs = tf.keras.layers.Input(shape=[100,100,1], dtype="float32") 

        down_stack = [
        downsample(32, 4, apply_batchnorm=False), # (bs, 50, 50, 32)
        downsample(64, 4), # (bs, 25, 25, 64)
        downsample(128, 4), # (bs, 13, 13, 128)
        downsample(256, 4), # (bs, 7, 7, 256)
        downsample(256, 4), # (bs, 4, 4, 256)
        downsample(256, 4), #(bs, 2, 2, 256)
        downsample(256, 4) #(bs, 1, 1, 256)

        ]

        up_stack = [
        upsample(256, 4, apply_dropout=True),
        upsample(256, 4, apply_dropout=True), # (bs, 2, 2, 256)
        upsample(256, 4), # (bs, 4, 4, 256)
        upsample(128, 4), # (bs, 7,  7, 128)
        upsample(64, 4), # (bs, 24, 24, 64)
        upsample(32, 4), # (bs, 25->50, 25->50, 32)
        ]

        initializer = tf.random_normal_initializer(0., 0.02)
        last = tf.keras.layers.Conv2DTranspose(OUTPUT_CHANNELS, 4,
                                             strides=2,
                                             padding='same',
                                             kernel_initializer=initializer, activation='tanh') # (bs, 128,128,1)

        x = inputs

      # Downsampling through the model
        skips = []
        for down in down_stack:
            x = down(x)
            skips.append(x)
    #         print('downsampling',x.shape)

        skips = reversed(skips[:-1])

      # Upsampling and establishing the skip connections
        for up, skip in zip(up_stack, skips):
            x = up(x)
            if x.shape[1]==8 or x.shape[1]==14 or x.shape[1]==26:
                x = x[:,:x.shape[1]-1,:x.shape[1]-1,:]
    #         print('upsampling',x.shape)
            x = tf.keras.layers.Concatenate()([x, skip])
    #         print(' after concat',x.shape)

        x = last(x)
        x = tf.cast(x, tf.float32)
        return tf.keras.Model(inputs=inputs, outputs=x)


    def Discriminator():
        initializer = tf.random_normal_initializer(0., 0.02)

        inp = tf.keras.layers.Input(shape=[100,100,1], name='input_image', dtype="float32")
        tar = tf.keras.layers.Input(shape=[100,100,1], name='target_image', dtype="float32")

        x = tf.keras.layers.concatenate([inp, tar]) # (bs, 100,100,channel*2=2)
        down1 = downsample(64, 4, False)(x) # (bs, 50,50, 64) batchnorm = False ?? 
        down2 = downsample(128, 4)(down1) # (bs, 25, 25, 128)

        zero_pad1 = tf.keras.layers.ZeroPadding2D()(down2) # (bs, 27, 27, 256)
        conv = tf.keras.layers.Conv2D(256, 4, strides=1,
                                    kernel_initializer=initializer,
                                    use_bias=False)(zero_pad1) # (bs, 27-4+1=24, 24, 256)

        batchnorm1 = tf.keras.layers.BatchNormalization()(conv)
        leaky_relu = tf.keras.layers.LeakyReLU()(batchnorm1)
        zero_pad2 = tf.keras.layers.ZeroPadding2D()(leaky_relu) # (bs, 26, 26, 256)
        last = tf.keras.layers.Conv2D(1, 4, strides=1,
                                    kernel_initializer=initializer)(zero_pad2) # (bs, 23, 23, 1)

        return tf.keras.Model(inputs=[inp, tar], outputs=last)

    generator = Generator()
    discriminator = Discriminator()

    generator.load_weights(r'F:\gan\model\mv\breast_xyz_model\g380_20200826-200239')
    discriminator.load_weights(r'F:\gan\model\mv\breast_xyz_model\d380_20200826-200240')

    return generator, discriminator