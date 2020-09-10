#!/usr/bin/env python
# coding: utf-8

# In[6]:


def evaluation(dis,save_path=None):
    from matplotlib import pyplot as plt
    import numpy as np
    import tensorflow as tf
    
    length =len(dis)
    title = ['input','target','origin','sagan','resU-net']
    
    
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
 
    
    plt.figure(figsize=(15,5))
    for i in range(length): # len(dis) = 3, 4, 5
        if i>=2:
            nmse_ele = np.square((dis[i]+1)-(dis[1]+1)).mean()
            rnmse_ele = 100*(np.sqrt(np.sum(np.square(dis[i]-dis[1]))/np.sum(np.square(dis[1]))))
            mse=rnmse_ele
            psnr=20*(np.log10(np.max(dis[i]+1)/nmse_ele))
            ssim=ssim_function((dis[i]+1),(dis[1]+1))
            ssim2=tf.image.ssim((dis[i]+1),(dis[1]+1),max_val=2)
            print('mse:{},psnr:{},ssim:{},ssim2:{}'.format(np.round(mse,3), np.round(psnr,3),np.round(ssim,3),np.round(ssim2.numpy(),3)))
                  
        plt.subplot(1,length,i+1)
        plt.imshow(dis[i][:,:,0],cmap='gray',vmin=-1,vmax=1)
        plt.title(title[i])
        
    if save_path:
        pt_ssim = '_{}.png'.format(np.round(ssim,2))
        plt.savefig(save_path+pt_ssim)
    plt.show()


    return np.round(mse,3), np.round(psnr,3),np.round(ssim,3),np.round(ssim2.numpy(),3)

def ssim_function(x,y):
    import numpy as np
    import tensorflow as tf
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


# In[ ]:




