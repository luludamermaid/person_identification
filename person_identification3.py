# -*- coding: utf-8 -*-
"""
Created on Tue May  5 13:51:28 2020

@author: heyru
"""
'''
File setup:
    - code repo at https://github.com/luludamermaid/person_identification
    - extract/copy all code files in a desired directory. Create a 'log' folder under this directory.
    	- The directory structure of facial image files should be as below:
    		|-Testing
    			|-name of person1
    			|-name of person2
    			|-name of person3
    		-|Training
    			|-name of person1
    			|-name of person2
    			|-name of person3
    		-|Cross_validation
    			|-name of person1
    			|-name of person2
    			|-name of person3
    	- The folder 'name of person1' , 'name of person2', 'name of person3'.. under 'Training' folder are for storing facial images for the person identification model to train, as part of learning, how to identify persons.  'name of person1' , 'name of person2', 'name of person3' would be named as the person's name  (for example, Jen, Jim,..). 
    	- The folder 'name of person1' , 'name of person2', 'name of person3'.. under 'Testing' folder are for storing facial images for the person identification model to test, as part of building the model.  'name of person1' , 'name of person2', 'name of person3' would be named as the person's name  (for example, Jen, Jim,..). 
    	- The folder 'name of person1' , 'name of person2', 'name of person3'.. under 'Cross_validation' folder are for storing facial images the user wants the model to identify persons based.
        
        
Anaconda Setup:
    - Bring up anaconda prompt, run these commands (referenced from this site https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/install.html)
        #create an environment with tensorflow and opencv. 
        
        conda create -n your_env_name pip python=3.8
        conda activate your_env_name
        pip install --ignore-installed --upgrade tensorflow==2.2.0
        
        #install opencv
        conda install -c conda-forge opencv   
    
        #install pillow for PIL 
        pip install pillow
    
        #install matplotlib
        pip install matplotlib
 

Spyder setup:
    - Bring up Anaconda (GUI). At Spyder, if there's no 'Launch' button, click on 'Install'. Then click on 'Launch' at Spyder.
    - at Spyder, open 'person_identification3.py'
    - At Spyder, at the upper right corner, change the working directory to the directory where person_identification3.py is.
    - Go to Tools>Preferences>Ipython Console>Graphics>Graphics Backend. Choose 'Inline' from the dropdown menu.
    - In order to show images on the console window when the program runs. At upper right pane in Spyder, you'll see 'Variable explorer', 'help', 'plots', 'files' tab. Choose 'Plots' tab, then click on the 3 horizontal lines icon at the upper right corner of the pane, uncheck 'Mute Inline Plotting'.
   

Run instructions (with Spyder on Anaconda):
    1. if desired, set variable values in the person_identification3.py code at 'user, please set variables here' section
    2. At Spyder, make sure you choose correct working directory in the top right corner of Spyder window. The working directory is the directory where this file is.
    3. If you want to see stats (accuracy and loss over epochs being run so you can tell how many epochs is enough to run), run Tensorboard by bringing up another Anaconda prompt, 
        activate your_environment
        cd your_directory_where_person_identification3.py_is
        tensorboard --logdir=logs --port=6006
        
        open a web browser, put in http://localhost:6006/ as an address. Tensorboard will show the stat graphs here.
    
    
Result:
Training: Noon 7, Kayne 5, Lu 5
Testing: Noon 6, Kayne 6, Lu 6
cv: Noon 6, Lu 2, Kayne 9
    1. with USE_CROPPED_PICS=1: only train, test and validate on cropped pics. The model gets 2 out of 13 images wrong. Predicting Lu20.jpg and Kayne30.jpg as Noon
    Epoch 15/15
8/8 [==============================] - 2s 266ms/step - loss: 1.5506e-06 - acc: 1.0000 - val_loss: 0.4067 - val_acc: 0.9167

    2. with USE_CROPPED_PICS=0: train, test, validate on regular pics. The model gets 3 out of 13 images wrong. 2 Kayne pics predicted as Noon, and 1 Kayne pic predicted as Lu
        Epoch 15/15
8/8 [==============================] - 3s 407ms/step - loss: 3.3528e-07 - acc: 1.0000 - val_loss: 0.8468 - val_acc: 0.8333

    3. 32 epochs, use cropped pics, 2 out of 13 images wrong. Noon and Lu predicted as Kayne
    Epoch 32/32
8/8 [==============================] - 3s 314ms/step - loss: 0.0000e+00 - acc: 1.0000 - val_loss: 0.2642 - val_acc: 0.9167

Training: Noon: 15, Kayne 16, Lu 10
Testing: Noon 2, Kayne 2, Lu 2
cv: Noon 1,Kayne 2, Lu 1
    1. 15 epochs, use_cropped_pics=1. wrong count =1
    
Training: 60% - KayneWest, JenAniston, BrunoMars, KevinHart: 24
Testing: 20% - KayneWest, JenAniston, BrunoMars, KevinHart: 12
cv: 20% - KayneWest, JenAniston, BrunoMars, KevinHart: 4
    1. 15 epochs, use_cropped_pics=0. wrong count = 8 (increasing neural networks from 512&4 to 1024&512&256&4 still yields the same result)
    Epoch 15/15
8/8 [==============================] - 8s 975ms/step - loss: 1.4322e-04 - acc: 1.0000 - val_loss: 2.7668 - val_acc: 0.5625
    2. same as 1, but use_cropped_pics=1. wrong count =6 out of 16
    Epoch 15/15
8/8 [==============================] - 3s 326ms/step - loss: 4.0249e-04 - acc: 1.0000 - val_loss: 0.2336 - val_acc: 0.9459
    3. same as 2 increase neural networks from 512&4 to 1024&512&256&4. wrong count=6 out of 16..
    4. change neural networks back to 512&4, remove JenAnniston. wrong couunt = 6 out of 12
    5. increase one more conv and filtering: wrong count=7
    
2 persons only:
1. use_cropped_pics=1. optimizer=RMSProp wrong count=1 on a false cropped pic
Epoch 15/15
8/8 [==============================] - 2s 291ms/step - loss: 6.9053e-05 - acc: 1.0000 - val_loss: 3.8743e-05 - val_acc: 1.0000    

2. sigmoid activation wrong count = 4, vs relu activation wrong count =2
-- optimizer experiments --
removed falsly cropped pic
1. RMSprop: wrong count =0, 2 (1st run, 2nd run,..)
2. Adam 1st wrong count = 0,0,0
3. SGD 1st run: wrong count =8
4. increase to 100 epochs, lr = 0.001. wrong count =0
5. increase lr to 0.01 for 100 epochs. wrong count = 15
6. decrease lr to 0.005 for 100 epochs. wrong count = 13
7. using tensorboard, convergence epochs = 20, so set it to 25, and batch size of 3. wrong count =0
8. batch size of 9 vs 3. wrong count = 1
  

'''
'''tweaks
- what's a dense layer, conv layer 
- decay rate
Conv2d(64 – change this number. layer size

'''
import os
from face_detection import crop_faces
import shutil
import datetime #for tensorboard

#user, please set variables here
USE_CROPPED_PICS=0 #set to 1 to train, test and cross validate on cropped pics
IMAGE_TARGET_SIZE=(300,300)
IMAGE_LEARNING_TARGET_SIZE=(150,150)
USE_LEARNING_RATE_SCHEDULE=0
EPOCHS = 40
LEARNING_RATE_NON_SCHEDULED = 0.001


dataset_path = "C:\\Lu\\AI\\classes\\Tensorflow_ML_DL\\LusTest\\whodat\\pics\\dataset"
model_path = "C:\\Lu\\AI\\classes\\Tensorflow_ML_DL\\LusTest\\whodat\\model"
training_folder_name = "Training"
testing_folder_name = "Testing"
cross_validation_folder_name = "Cross_validation"

#end of user variable settings


training_dir = os.path.join(dataset_path,training_folder_name)
testing_dir = os.path.join(dataset_path, testing_folder_name)
cross_validation_dir =os.path.join(dataset_path, cross_validation_folder_name)
person_names=os.listdir(training_dir)
persons = len(person_names)

if USE_CROPPED_PICS:
    #create a new tree
    cropped_dataset_path = os.path.join(dataset_path,"cropped")
    if os.path.exists(cropped_dataset_path):
        shutil.rmtree(cropped_dataset_path)
    os.makedirs(cropped_dataset_path)
    training_dir = os.path.join(cropped_dataset_path,training_folder_name)
    testing_dir = os.path.join(cropped_dataset_path, testing_folder_name)
    cross_validation_dir = os.path.join(cropped_dataset_path, cross_validation_folder_name)
    
    #go into each training dir and crop, and copy to the new path
    for name in person_names:
        #training pics
        training_input_face_path = os.path.join(dataset_path,training_folder_name,name)
        training_cropped_face_path = os.path.join(training_dir, name)
        crop_faces(training_input_face_path, training_cropped_face_path,IMAGE_TARGET_SIZE[0],IMAGE_TARGET_SIZE[1])
        
        #testing pics
        testing_input_face_path = os.path.join(dataset_path,testing_folder_name,name)
        testing_cropped_face_path = os.path.join(testing_dir, name)
        crop_faces(testing_input_face_path, testing_cropped_face_path,IMAGE_TARGET_SIZE[0],IMAGE_TARGET_SIZE[1])
        
        #cross_validation pics
        cross_validation_input_face_path = os.path.join(dataset_path,cross_validation_folder_name,name)
        cross_validation_cropped_face_path = os.path.join(cross_validation_dir, name)
        crop_faces(cross_validation_input_face_path, cross_validation_cropped_face_path,IMAGE_TARGET_SIZE[0],IMAGE_TARGET_SIZE[1])
        
#Note that because we are facing a two-class classification problem, i.e. a binary classification problem, we will end our network with a sigmoid activation, so that the output of our network will be a single scalar between 0 and 1, encoding the probability that the current image is class 1 (as opposed to class 0).

import tensorflow as tf
if USE_LEARNING_RATE_SCHEDULE == 1:
    logdir="logs\\scalars\\" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    file_writer = tf.summary.create_file_writer(logdir+"\\metrics")
    file_writer.set_as_default()
    
    def lr_schedule(epoch):
        learning_rate = 0.01
        if epoch > 10:
            learning_rate =0.007
        if epoch > 20:
            learning_rate = 0.005
        if epoch > 30:
            learning_rate =0.001
        tf.summary.scalar('learning rate', learning_rate, step=epoch)
        return learning_rate
            
    lr_callback = tf.keras.callbacks.LearningRateScheduler(lr_schedule) 
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)  
    model_callbacks = [tensorboard_callback, lr_callback]


model = tf.keras.models.Sequential([
    # Note the input shape is the desired size of the image 300x300 with 3 bytes color
    # This is the first convolution
    tf.keras.layers.Conv2D(16, (3,3), activation='relu', input_shape=(IMAGE_LEARNING_TARGET_SIZE[0], IMAGE_LEARNING_TARGET_SIZE[1], 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    # The second convolution
    tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    # The third convolution
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    # The fourth convolution
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    # The fifth convolution
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    # Flatten the results to feed into a DNN
    tf.keras.layers.Flatten(),
    # 512 neuron hidden layer
    #relu - linear rectifier function. relu is easier to train and often achieves better performance. relu overcomes the vanishing gradient problem, allowing models to learn faster and perform better.
    tf.keras.layers.Dense(512, activation='relu'),
    
    #last layer is how many categories you have. use softmax because it gives probability based on the output number of the last neuron layer.
    tf.keras.layers.Dense(persons, activation='softmax')
    ])
    
model.summary()
    
#Next, we'll configure the specifications for model training. We will train our model with the binary_crossentropy loss, because it's a binary classification problem and our final activation is a sigmoid. (For a refresher on loss metrics, see the Machine Learning Crash Course.) We will use the rmsprop optimizer with a learning rate of 0.001. During training, we will want to monitor classification accuracy.
#NOTE: In this case, using the RMSprop optimization algorithm is preferable to stochastic gradient descent (SGD), because RMSprop automates learning-rate tuning for us. (Other optimizers, such as Adam and Adagrad, also automatically adapt the learning rate during training, and would work equally well here.)


model.compile(loss= 'categorical_crossentropy',
          optimizer=tf.keras.optimizers.Adam(lr=LEARNING_RATE_NON_SCHEDULED),
          metrics=['accuracy'])

if USE_LEARNING_RATE_SCHEDULE ==0 :
    log_dir = "logs\\" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    model_callbacks = [tensorboard_callback]


from tensorflow.keras.preprocessing.image import ImageDataGenerator
#normalizing the pixel values to be in the [0, 1] range (originally all values are in the [0, 255] range).
# All images will be rescaled by 1./255
train_datagen = ImageDataGenerator(rescale=1/255)
validation_datagen = ImageDataGenerator(rescale=1/255)

# Flow training images in batches of 128 using train_datagen generator
train_generator = train_datagen.flow_from_directory(
        training_dir,  # This is the source directory for training images
        target_size=(IMAGE_LEARNING_TARGET_SIZE[0], IMAGE_LEARNING_TARGET_SIZE[1]), #resize images to 150x150 so it's easier to process
        #batch_size=128,
        # Since we use binary_crossentropy loss, we need binary labels
        class_mode='categorical')

validation_generator = validation_datagen.flow_from_directory(
        testing_dir,
        target_size =(IMAGE_LEARNING_TARGET_SIZE[0],IMAGE_LEARNING_TARGET_SIZE[1]),
        #batch_size = 32,
        class_mode = 'categorical')
#at this point train_generator is of DataFrameIterator class containing input (x) of 128x300x300x3 matrix, and output (y) of vector of number of labels associating with the inputs(x, aka the i/p images) 
#A DataFrameIterator yielding tuples of (x, y) where x is a numpy array containing a batch of images with shape (batch_size, *target_size, channels) and y is a numpy array of corresponding labels.
#here, batch_size is 128, target_size is 300x300, channels is 3 (RGB = 3 bytes/pixel)
#for "grayscale" channel =1, "rgb" channel=3, "rgba" channel=4. Default channel is "rgb". 
history = model.fit(
      train_generator,
      steps_per_epoch=3, #was 8  
      epochs=EPOCHS,        
      verbose=1,
      validation_data = validation_generator,
      validation_steps=3, #was 8
      callbacks=model_callbacks)
      



#saving model
#model.save(model_path)

#load model
#model = keras.models.load_model('path/to/location')

import numpy as np
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

wrongs=0
rights=0

for person in person_names:
    path = os.path.join(cross_validation_dir, person)
    for pic in os.listdir(path):
        picPath=os.path.join(path,pic)
        img = image.load_img(picPath, target_size=(IMAGE_LEARNING_TARGET_SIZE[0], IMAGE_LEARNING_TARGET_SIZE[1]))
        x = image.img_to_array(img)
        #now x.shape is (300,300,3)
        x = np.expand_dims(x, axis=0)
        #now x.shape is (1,300,300,3)
        
        images = np.vstack([x])
        #print("images shape after vstack= "%images.shape)
        
        classes = model.predict(images, batch_size=10)
        for i in range(persons):
            if classes[0,i]>0.5:
                print("predicted %s. Confidence %.2f%%"%(person_names[i],100*classes[0,i]))
                if person_names[i] != person:
                    print("*****WRONG***** %s expected"%(person))
                    wrongs +=1
                else:
                    rights +=1
        img = mpimg.imread(picPath)
        plt.imshow(img)
        plt.show()
            
            
     
print("wrong predictions = %d, correct predictions = %d"%(wrongs, rights))


#clear session

import gc
tf.keras.backend.clear_session()
gc.collect()
del model
