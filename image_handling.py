import numpy as np
import matplotlib.pyplot as plt # for plotting

def imshow(img):
  ''' function to show image '''
  img = img / 2 + 0.5 # unnormalize
  npimg = img.numpy() # convert to numpy objects

  fig1 = plt.figure()

  plt.imshow(np.transpose(npimg, (1, 2, 0)))
  #plt.show(block=False)
  #plt.ioff()
  fig1.show()