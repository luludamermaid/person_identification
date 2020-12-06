About the program:
This program is for identifying persons through facial images. 
At the development time, the program was running on Spyder IDE in Anaconda.

References:
- face_detection.py is obtained from https://medium.com/@somaniswastik/face-recognition-using-tensorflow-pre-trained-model-opencv-91184efa4aaf. I

- Face detection config files (haarcascade_xxx.xml files) are obtained from https://github.com/opencv/opencv/tree/master/data/haarcascades

Setup Instructions:
1. File setup
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
	
2. Anaconda setup
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
3. Spyder setup
    - Bring up Anaconda (GUI). At Spyder, if there's no 'Launch' button, click on 'Install'. Then click on 'Launch' at Spyder.
	- at Spyder, open 'person_identification3.py'
    - At Spyder, at the upper right corner, change the working directory to the directory where person_identification3.py is.
    - Go to Tools>Preferences>Ipython Console>Graphics>Graphics Backend. Choose 'Inline' from the dropdown menu.
    - In order to show images on the console window when the program runs. At upper right pane in Spyder, you'll see 'Variable explorer', 'help', 'plots', 'files' tab. Choose 'Plots' tab, then click on the 3 horizontal lines icon at the upper right corner of the pane, uncheck 'Mute Inline Plotting'.

Run instructions
    1. if desired, set variable values in the person_identification3.py code at 'user, please set variables here' section
    2. At Spyder, make sure you choose correct working directory in the top right corner of Spyder window. The working directory is the directory where this file is.
    3. If you want to see stats (accuracy and loss over epochs being run so you can tell how many epochs is enough to run), run Tensorboard by bringing up another Anaconda prompt, 
        activate your_environment
        cd your_directory_where_person_identification3.py_is
        tensorboard --logdir=logs --port=6006
        
        open a web browser, put in http://localhost:6006/ as an address. Tensorboard will show the stat graphs here.


Run Result example:
	- After running in Spyder, the Console pane would show something like below. At the end the program shows summary of correct and wrong predictions.

	Note that the console should show the facial images, but it doesn't show here because the readme file is a text only file.
	
	
	Python 3.8.5 (default, Sep  3 2020, 21:29:08) [MSC v.1916 64 bit (AMD64)]
Type "copyright", "credits" or "license" for more information.

IPython 7.19.0 -- An enhanced Interactive Python.

In [1]: runfile('C:/Lu/AI/classes/Tensorflow_ML_DL/LusTest/whodat/whodat/person_identification3.py', wdir='C:/Lu/AI/classes/Tensorflow_ML_DL/LusTest/whodat/whodat')

2020-12-06 10:35:59.926637: W tensorflow/stream_executor/platform/default/dso_loader.cc:55] Could not load dynamic library 'cudart64_101.dll'; dlerror: cudart64_101.dll not found
2020-12-06 10:35:59.927028: I tensorflow/stream_executor/cuda/cudart_stub.cc:29] Ignore above cudart dlerror if you do not have a GPU set up on your machine.
Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
conv2d (Conv2D)              (None, 148, 148, 16)      448       
_________________________________________________________________
max_pooling2d (MaxPooling2D) (None, 74, 74, 16)        0         
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 72, 72, 32)        4640      
_________________________________________________________________
max_pooling2d_1 (MaxPooling2 (None, 36, 36, 32)        0         
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 34, 34, 64)        18496     
_________________________________________________________________
max_pooling2d_2 (MaxPooling2 (None, 17, 17, 64)        0         
_________________________________________________________________
conv2d_3 (Conv2D)            (None, 15, 15, 64)        36928     
_________________________________________________________________
max_pooling2d_3 (MaxPooling2 (None, 7, 7, 64)          0         
_________________________________________________________________
conv2d_4 (Conv2D)            (None, 5, 5, 64)          36928     
_________________________________________________________________
max_pooling2d_4 (MaxPooling2 (None, 2, 2, 64)          0         
_________________________________________________________________
flatten (Flatten)            (None, 256)               0         
_________________________________________________________________
dense (Dense)                (None, 512)               131584    
_________________________________________________________________
dense_1 (Dense)              (None, 3)                 1539      
=================================================================
Total params: 230,563
Trainable params: 230,563
Non-trainable params: 0
_________________________________________________________________
Found 95 images belonging to 3 classes.
Found 31 images belonging to 3 classes.

2020-12-06 10:35:59.926637: W tensorflow/stream_executor/platform/default/dso_loader.cc:55] Could not load dynamic library 'cudart64_101.dll'; dlerror: cudart64_101.dll not found
2020-12-06 10:35:59.927028: I tensorflow/stream_executor/cuda/cudart_stub.cc:29] Ignore above cudart dlerror if you do not have a GPU set up on your machine.
2020-12-06 10:36:05.053606: W tensorflow/stream_executor/platform/default/dso_loader.cc:55] Could not load dynamic library 'nvcuda.dll'; dlerror: nvcuda.dll not found
2020-12-06 10:36:05.053656: E tensorflow/stream_executor/cuda/cuda_driver.cc:313] failed call to cuInit: UNKNOWN ERROR (303)
2020-12-06 10:36:05.057783: I tensorflow/stream_executor/cuda/cuda_diagnostics.cc:169] retrieving CUDA diagnostic information for host: LAPTOP-ND1S8BQN
2020-12-06 10:36:05.057869: I tensorflow/stream_executor/cuda/cuda_diagnostics.cc:176] hostname: LAPTOP-ND1S8BQN
2020-12-06 10:36:05.060562: I tensorflow/core/platform/cpu_feature_guard.cc:143] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2
2020-12-06 10:36:05.095928: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x1610de00480 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-12-06 10:36:05.095958: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
2020-12-06 10:36:05.241325: I tensorflow/core/profiler/lib/profiler_session.cc:159] Profiler session started.
Epoch 1/40
1/3 [=========>....................] - ETA: 0s - loss: 1.0949 - accuracy: 0.5938
2020-12-06 10:35:59.926637: W tensorflow/stream_executor/platform/default/dso_loader.cc:55] Could not load dynamic library 'cudart64_101.dll'; dlerror: cudart64_101.dll not found
2020-12-06 10:35:59.927028: I tensorflow/stream_executor/cuda/cudart_stub.cc:29] Ignore above cudart dlerror if you do not have a GPU set up on your machine.
2020-12-06 10:36:05.053606: W tensorflow/stream_executor/platform/default/dso_loader.cc:55] Could not load dynamic library 'nvcuda.dll'; dlerror: nvcuda.dll not found
2020-12-06 10:36:05.053656: E tensorflow/stream_executor/cuda/cuda_driver.cc:313] failed call to cuInit: UNKNOWN ERROR (303)
2020-12-06 10:36:05.057783: I tensorflow/stream_executor/cuda/cuda_diagnostics.cc:169] retrieving CUDA diagnostic information for host: LAPTOP-ND1S8BQN
2020-12-06 10:36:05.057869: I tensorflow/stream_executor/cuda/cuda_diagnostics.cc:176] hostname: LAPTOP-ND1S8BQN
2020-12-06 10:36:05.060562: I tensorflow/core/platform/cpu_feature_guard.cc:143] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2
2020-12-06 10:36:05.095928: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x1610de00480 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-12-06 10:36:05.095958: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
2020-12-06 10:36:05.241325: I tensorflow/core/profiler/lib/profiler_session.cc:159] Profiler session started.
2020-12-06 10:36:07.669930: I tensorflow/core/profiler/lib/profiler_session.cc:159] Profiler session started.
2/3 [===================>..........] - ETA: 0s - loss: 1.0359 - accuracy: 0.6508
2020-12-06 10:35:59.926637: W tensorflow/stream_executor/platform/default/dso_loader.cc:55] Could not load dynamic library 'cudart64_101.dll'; dlerror: cudart64_101.dll not found
2020-12-06 10:35:59.927028: I tensorflow/stream_executor/cuda/cudart_stub.cc:29] Ignore above cudart dlerror if you do not have a GPU set up on your machine.
2020-12-06 10:36:05.053606: W tensorflow/stream_executor/platform/default/dso_loader.cc:55] Could not load dynamic library 'nvcuda.dll'; dlerror: nvcuda.dll not found
2020-12-06 10:36:05.053656: E tensorflow/stream_executor/cuda/cuda_driver.cc:313] failed call to cuInit: UNKNOWN ERROR (303)
2020-12-06 10:36:05.057783: I tensorflow/stream_executor/cuda/cuda_diagnostics.cc:169] retrieving CUDA diagnostic information for host: LAPTOP-ND1S8BQN
2020-12-06 10:36:05.057869: I tensorflow/stream_executor/cuda/cuda_diagnostics.cc:176] hostname: LAPTOP-ND1S8BQN
2020-12-06 10:36:05.060562: I tensorflow/core/platform/cpu_feature_guard.cc:143] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2
2020-12-06 10:36:05.095928: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x1610de00480 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-12-06 10:36:05.095958: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
2020-12-06 10:36:05.241325: I tensorflow/core/profiler/lib/profiler_session.cc:159] Profiler session started.
2020-12-06 10:36:07.669930: I tensorflow/core/profiler/lib/profiler_session.cc:159] Profiler session started.
2020-12-06 10:36:08.336666: I tensorflow/core/profiler/rpc/client/save_profile.cc:168] Creating directory: logs\20201206-103605\train\plugins\profile\2020_12_06_00_36_08
2020-12-06 10:36:08.339279: I tensorflow/core/profiler/rpc/client/save_profile.cc:174] Dumped gzipped tool data for trace.json.gz to logs\20201206-103605\train\plugins\profile\2020_12_06_00_36_08\LAPTOP-ND1S8BQN.trace.json.gz
2020-12-06 10:36:08.342740: I tensorflow/core/profiler/utils/event_span.cc:288] Generation of step-events took 0.023 ms

2020-12-06 10:36:08.351785: I tensorflow/python/profiler/internal/profiler_wrapper.cc:87] Creating directory: logs\20201206-103605\train\plugins\profile\2020_12_06_00_36_08Dumped tool data for overview_page.pb to logs\20201206-103605\train\plugins\profile\2020_12_06_00_36_08\LAPTOP-ND1S8BQN.overview_page.pb
Dumped tool data for input_pipeline.pb to logs\20201206-103605\train\plugins\profile\2020_12_06_00_36_08\LAPTOP-ND1S8BQN.input_pipeline.pb
Dumped tool data for tensorflow_stats.pb to logs\20201206-103605\train\plugins\profile\2020_12_06_00_36_08\LAPTOP-ND1S8BQN.tensorflow_stats.pb
Dumped tool data for kernel_stats.pb to logs\20201206-103605\train\plugins\profile\2020_12_06_00_36_08\LAPTOP-ND1S8BQN.kernel_stats.pb

3/3 [==============================] - ETA: 0s - loss: 0.9796 - accuracy: 0.6632WARNING:tensorflow:Your input ran out of data; interrupting training. Make sure that your dataset or generator can generate at least `steps_per_epoch * epochs` batches (in this case, 3 batches). You may need to use the repeat() function when building your dataset.
3/3 [==============================] - 3s 940ms/step - loss: 0.9796 - accuracy: 0.6632 - val_loss: 0.9580 - val_accuracy: 0.6452
Epoch 2/40
3/3 [==============================] - ETA: 0s - loss: 0.7893 - accuracy: 0.7158WARNING:tensorflow:Your input ran out of data; interrupting training. Make sure that your dataset or generator can generate at least `steps_per_epoch * epochs` batches (in this case, 3 batches). You may need to use the repeat() function when building your dataset.
3/3 [==============================] - 2s 656ms/step - loss: 0.7893 - accuracy: 0.7158 - val_loss: 0.8932 - val_accuracy: 0.6452
Epoch 3/40
3/3 [==============================] - ETA: 0s - loss: 0.7491 - accuracy: 0.7158WARNING:tensorflow:Your input ran out of data; interrupting training. Make sure that your dataset or generator can generate at least `steps_per_epoch * epochs` batches (in this case, 3 batches). You may need to use the repeat() function when building your dataset.
3/3 [==============================] - 2s 666ms/step - loss: 0.7491 - accuracy: 0.7158 - val_loss: 0.8674 - val_accuracy: 0.6452
Epoch 4/40
3/3 [==============================] - ETA: 0s - loss: 0.7244 - accuracy: 0.7158WARNING:tensorflow:Your input ran out of data; interrupting training. Make sure that your dataset or generator can generate at least `steps_per_epoch * epochs` batches (in this case, 3 batches). You may need to use the repeat() function when building your dataset.
3/3 [==============================] - 2s 670ms/step - loss: 0.7244 - accuracy: 0.7158 - val_loss: 0.8916 - val_accuracy: 0.6452
Epoch 5/40
3/3 [==============================] - ETA: 0s - loss: 0.6692 - accuracy: 0.7158WARNING:tensorflow:Your input ran out of data; interrupting training. Make sure that your dataset or generator can generate at least `steps_per_epoch * epochs` batches (in this case, 3 batches). You may need to use the repeat() function when building your dataset.
3/3 [==============================] - 2s 677ms/step - loss: 0.6692 - accuracy: 0.7158 - val_loss: 0.7674 - val_accuracy: 0.6774
Epoch 6/40
3/3 [==============================] - ETA: 0s - loss: 0.6193 - accuracy: 0.7474WARNING:tensorflow:Your input ran out of data; interrupting training. Make sure that your dataset or generator can generate at least `steps_per_epoch * epochs` batches (in this case, 3 batches). You may need to use the repeat() function when building your dataset.
3/3 [==============================] - 2s 667ms/step - loss: 0.6193 - accuracy: 0.7474 - val_loss: 0.7681 - val_accuracy: 0.6774
Epoch 7/40
3/3 [==============================] - ETA: 0s - loss: 0.5236 - accuracy: 0.7789WARNING:tensorflow:Your input ran out of data; interrupting training. Make sure that your dataset or generator can generate at least `steps_per_epoch * epochs` batches (in this case, 3 batches). You may need to use the repeat() function when building your dataset.
3/3 [==============================] - 2s 670ms/step - loss: 0.5236 - accuracy: 0.7789 - val_loss: 0.6007 - val_accuracy: 0.8387
Epoch 8/40
3/3 [==============================] - ETA: 0s - loss: 0.4027 - accuracy: 0.8947WARNING:tensorflow:Your input ran out of data; interrupting training. Make sure that your dataset or generator can generate at least `steps_per_epoch * epochs` batches (in this case, 3 batches). You may need to use the repeat() function when building your dataset.
3/3 [==============================] - 2s 675ms/step - loss: 0.4027 - accuracy: 0.8947 - val_loss: 0.4953 - val_accuracy: 0.8387
Epoch 9/40
3/3 [==============================] - ETA: 0s - loss: 0.2952 - accuracy: 0.8842WARNING:tensorflow:Your input ran out of data; interrupting training. Make sure that your dataset or generator can generate at least `steps_per_epoch * epochs` batches (in this case, 3 batches). You may need to use the repeat() function when building your dataset.
3/3 [==============================] - 2s 653ms/step - loss: 0.2952 - accuracy: 0.8842 - val_loss: 0.3363 - val_accuracy: 0.8710
Epoch 10/40
3/3 [==============================] - ETA: 0s - loss: 0.2326 - accuracy: 0.9053WARNING:tensorflow:Your input ran out of data; interrupting training. Make sure that your dataset or generator can generate at least `steps_per_epoch * epochs` batches (in this case, 3 batches). You may need to use the repeat() function when building your dataset.
3/3 [==============================] - 2s 673ms/step - loss: 0.2326 - accuracy: 0.9053 - val_loss: 0.5887 - val_accuracy: 0.8387
Epoch 11/40
3/3 [==============================] - ETA: 0s - loss: 0.2129 - accuracy: 0.9158WARNING:tensorflow:Your input ran out of data; interrupting training. Make sure that your dataset or generator can generate at least `steps_per_epoch * epochs` batches (in this case, 3 batches). You may need to use the repeat() function when building your dataset.
3/3 [==============================] - 2s 668ms/step - loss: 0.2129 - accuracy: 0.9158 - val_loss: 0.3030 - val_accuracy: 0.8710
Epoch 12/40
3/3 [==============================] - ETA: 0s - loss: 0.1330 - accuracy: 0.9474WARNING:tensorflow:Your input ran out of data; interrupting training. Make sure that your dataset or generator can generate at least `steps_per_epoch * epochs` batches (in this case, 3 batches). You may need to use the repeat() function when building your dataset.
3/3 [==============================] - 2s 682ms/step - loss: 0.1330 - accuracy: 0.9474 - val_loss: 0.4359 - val_accuracy: 0.8710
Epoch 13/40
3/3 [==============================] - ETA: 0s - loss: 0.1139 - accuracy: 0.9684WARNING:tensorflow:Your input ran out of data; interrupting training. Make sure that your dataset or generator can generate at least `steps_per_epoch * epochs` batches (in this case, 3 batches). You may need to use the repeat() function when building your dataset.
3/3 [==============================] - 2s 665ms/step - loss: 0.1139 - accuracy: 0.9684 - val_loss: 0.2390 - val_accuracy: 0.9032
Epoch 14/40
3/3 [==============================] - ETA: 0s - loss: 0.0836 - accuracy: 0.9579WARNING:tensorflow:Your input ran out of data; interrupting training. Make sure that your dataset or generator can generate at least `steps_per_epoch * epochs` batches (in this case, 3 batches). You may need to use the repeat() function when building your dataset.
3/3 [==============================] - 2s 673ms/step - loss: 0.0836 - accuracy: 0.9579 - val_loss: 0.6026 - val_accuracy: 0.8710
Epoch 15/40
3/3 [==============================] - ETA: 0s - loss: 0.0511 - accuracy: 0.9895WARNING:tensorflow:Your input ran out of data; interrupting training. Make sure that your dataset or generator can generate at least `steps_per_epoch * epochs` batches (in this case, 3 batches). You may need to use the repeat() function when building your dataset.
3/3 [==============================] - 2s 660ms/step - loss: 0.0511 - accuracy: 0.9895 - val_loss: 0.1800 - val_accuracy: 0.9032
Epoch 16/40
3/3 [==============================] - ETA: 0s - loss: 0.0398 - accuracy: 0.9895WARNING:tensorflow:Your input ran out of data; interrupting training. Make sure that your dataset or generator can generate at least `steps_per_epoch * epochs` batches (in this case, 3 batches). You may need to use the repeat() function when building your dataset.
3/3 [==============================] - 2s 667ms/step - loss: 0.0398 - accuracy: 0.9895 - val_loss: 0.2132 - val_accuracy: 0.9032
Epoch 17/40
3/3 [==============================] - ETA: 0s - loss: 0.0243 - accuracy: 1.0000WARNING:tensorflow:Your input ran out of data; interrupting training. Make sure that your dataset or generator can generate at least `steps_per_epoch * epochs` batches (in this case, 3 batches). You may need to use the repeat() function when building your dataset.
3/3 [==============================] - 2s 691ms/step - loss: 0.0243 - accuracy: 1.0000 - val_loss: 0.4837 - val_accuracy: 0.9032
Epoch 18/40
3/3 [==============================] - ETA: 0s - loss: 0.0111 - accuracy: 1.0000WARNING:tensorflow:Your input ran out of data; interrupting training. Make sure that your dataset or generator can generate at least `steps_per_epoch * epochs` batches (in this case, 3 batches). You may need to use the repeat() function when building your dataset.
3/3 [==============================] - 2s 679ms/step - loss: 0.0111 - accuracy: 1.0000 - val_loss: 0.4287 - val_accuracy: 0.9032
Epoch 19/40
3/3 [==============================] - ETA: 0s - loss: 0.0088 - accuracy: 1.0000WARNING:tensorflow:Your input ran out of data; interrupting training. Make sure that your dataset or generator can generate at least `steps_per_epoch * epochs` batches (in this case, 3 batches). You may need to use the repeat() function when building your dataset.
3/3 [==============================] - 2s 656ms/step - loss: 0.0088 - accuracy: 1.0000 - val_loss: 0.3593 - val_accuracy: 0.9032
Epoch 20/40
3/3 [==============================] - ETA: 0s - loss: 0.0079 - accuracy: 1.0000WARNING:tensorflow:Your input ran out of data; interrupting training. Make sure that your dataset or generator can generate at least `steps_per_epoch * epochs` batches (in this case, 3 batches). You may need to use the repeat() function when building your dataset.
3/3 [==============================] - 2s 674ms/step - loss: 0.0079 - accuracy: 1.0000 - val_loss: 0.5725 - val_accuracy: 0.9032
Epoch 21/40
3/3 [==============================] - ETA: 0s - loss: 0.0036 - accuracy: 1.0000WARNING:tensorflow:Your input ran out of data; interrupting training. Make sure that your dataset or generator can generate at least `steps_per_epoch * epochs` batches (in this case, 3 batches). You may need to use the repeat() function when building your dataset.
3/3 [==============================] - 2s 674ms/step - loss: 0.0036 - accuracy: 1.0000 - val_loss: 0.7431 - val_accuracy: 0.9032
Epoch 22/40
3/3 [==============================] - ETA: 0s - loss: 0.0035 - accuracy: 1.0000WARNING:tensorflow:Your input ran out of data; interrupting training. Make sure that your dataset or generator can generate at least `steps_per_epoch * epochs` batches (in this case, 3 batches). You may need to use the repeat() function when building your dataset.
3/3 [==============================] - 2s 656ms/step - loss: 0.0035 - accuracy: 1.0000 - val_loss: 0.5043 - val_accuracy: 0.9032
Epoch 23/40
3/3 [==============================] - ETA: 0s - loss: 0.0015 - accuracy: 1.0000WARNING:tensorflow:Your input ran out of data; interrupting training. Make sure that your dataset or generator can generate at least `steps_per_epoch * epochs` batches (in this case, 3 batches). You may need to use the repeat() function when building your dataset.
3/3 [==============================] - 2s 670ms/step - loss: 0.0015 - accuracy: 1.0000 - val_loss: 0.3677 - val_accuracy: 0.9032
Epoch 24/40
3/3 [==============================] - ETA: 0s - loss: 0.0012 - accuracy: 1.0000    WARNING:tensorflow:Your input ran out of data; interrupting training. Make sure that your dataset or generator can generate at least `steps_per_epoch * epochs` batches (in this case, 3 batches). You may need to use the repeat() function when building your dataset.
3/3 [==============================] - 2s 680ms/step - loss: 0.0012 - accuracy: 1.0000 - val_loss: 0.3967 - val_accuracy: 0.9032
Epoch 25/40
3/3 [==============================] - ETA: 0s - loss: 6.0504e-04 - accuracy: 1.0000WARNING:tensorflow:Your input ran out of data; interrupting training. Make sure that your dataset or generator can generate at least `steps_per_epoch * epochs` batches (in this case, 3 batches). You may need to use the repeat() function when building your dataset.
3/3 [==============================] - 2s 692ms/step - loss: 6.0504e-04 - accuracy: 1.0000 - val_loss: 0.5181 - val_accuracy: 0.9032
Epoch 26/40
3/3 [==============================] - ETA: 0s - loss: 4.8459e-04 - accuracy: 1.0000WARNING:tensorflow:Your input ran out of data; interrupting training. Make sure that your dataset or generator can generate at least `steps_per_epoch * epochs` batches (in this case, 3 batches). You may need to use the repeat() function when building your dataset.
3/3 [==============================] - 2s 685ms/step - loss: 4.8459e-04 - accuracy: 1.0000 - val_loss: 0.6238 - val_accuracy: 0.9032
Epoch 27/40
3/3 [==============================] - ETA: 0s - loss: 6.2334e-04 - accuracy: 1.0000WARNING:tensorflow:Your input ran out of data; interrupting training. Make sure that your dataset or generator can generate at least `steps_per_epoch * epochs` batches (in this case, 3 batches). You may need to use the repeat() function when building your dataset.
3/3 [==============================] - 2s 682ms/step - loss: 6.2334e-04 - accuracy: 1.0000 - val_loss: 0.6518 - val_accuracy: 0.9032
Epoch 28/40
3/3 [==============================] - ETA: 0s - loss: 3.0870e-04 - accuracy: 1.0000WARNING:tensorflow:Your input ran out of data; interrupting training. Make sure that your dataset or generator can generate at least `steps_per_epoch * epochs` batches (in this case, 3 batches). You may need to use the repeat() function when building your dataset.
3/3 [==============================] - 2s 695ms/step - loss: 3.0870e-04 - accuracy: 1.0000 - val_loss: 0.6404 - val_accuracy: 0.9032
Epoch 29/40
3/3 [==============================] - ETA: 0s - loss: 1.7665e-04 - accuracy: 1.0000WARNING:tensorflow:Your input ran out of data; interrupting training. Make sure that your dataset or generator can generate at least `steps_per_epoch * epochs` batches (in this case, 3 batches). You may need to use the repeat() function when building your dataset.
3/3 [==============================] - 2s 669ms/step - loss: 1.7665e-04 - accuracy: 1.0000 - val_loss: 0.6273 - val_accuracy: 0.9032
Epoch 30/40
3/3 [==============================] - ETA: 0s - loss: 1.4423e-04 - accuracy: 1.0000WARNING:tensorflow:Your input ran out of data; interrupting training. Make sure that your dataset or generator can generate at least `steps_per_epoch * epochs` batches (in this case, 3 batches). You may need to use the repeat() function when building your dataset.
3/3 [==============================] - 2s 694ms/step - loss: 1.4423e-04 - accuracy: 1.0000 - val_loss: 0.6120 - val_accuracy: 0.9032
Epoch 31/40
3/3 [==============================] - ETA: 0s - loss: 1.2930e-04 - accuracy: 1.0000WARNING:tensorflow:Your input ran out of data; interrupting training. Make sure that your dataset or generator can generate at least `steps_per_epoch * epochs` batches (in this case, 3 batches). You may need to use the repeat() function when building your dataset.
3/3 [==============================] - 2s 685ms/step - loss: 1.2930e-04 - accuracy: 1.0000 - val_loss: 0.6033 - val_accuracy: 0.9032
Epoch 32/40
3/3 [==============================] - ETA: 0s - loss: 1.3548e-04 - accuracy: 1.0000WARNING:tensorflow:Your input ran out of data; interrupting training. Make sure that your dataset or generator can generate at least `steps_per_epoch * epochs` batches (in this case, 3 batches). You may need to use the repeat() function when building your dataset.
3/3 [==============================] - 2s 671ms/step - loss: 1.3548e-04 - accuracy: 1.0000 - val_loss: 0.6007 - val_accuracy: 0.9032
Epoch 33/40
3/3 [==============================] - ETA: 0s - loss: 1.3054e-04 - accuracy: 1.0000WARNING:tensorflow:Your input ran out of data; interrupting training. Make sure that your dataset or generator can generate at least `steps_per_epoch * epochs` batches (in this case, 3 batches). You may need to use the repeat() function when building your dataset.
3/3 [==============================] - 2s 677ms/step - loss: 1.3054e-04 - accuracy: 1.0000 - val_loss: 0.6088 - val_accuracy: 0.9032
Epoch 34/40
3/3 [==============================] - ETA: 0s - loss: 1.1948e-04 - accuracy: 1.0000WARNING:tensorflow:Your input ran out of data; interrupting training. Make sure that your dataset or generator can generate at least `steps_per_epoch * epochs` batches (in this case, 3 batches). You may need to use the repeat() function when building your dataset.
3/3 [==============================] - 2s 685ms/step - loss: 1.1948e-04 - accuracy: 1.0000 - val_loss: 0.6217 - val_accuracy: 0.9032
Epoch 35/40
3/3 [==============================] - ETA: 0s - loss: 1.0973e-04 - accuracy: 1.0000WARNING:tensorflow:Your input ran out of data; interrupting training. Make sure that your dataset or generator can generate at least `steps_per_epoch * epochs` batches (in this case, 3 batches). You may need to use the repeat() function when building your dataset.
3/3 [==============================] - 2s 658ms/step - loss: 1.0973e-04 - accuracy: 1.0000 - val_loss: 0.6388 - val_accuracy: 0.9032
Epoch 36/40
3/3 [==============================] - ETA: 0s - loss: 9.8459e-05 - accuracy: 1.0000WARNING:tensorflow:Your input ran out of data; interrupting training. Make sure that your dataset or generator can generate at least `steps_per_epoch * epochs` batches (in this case, 3 batches). You may need to use the repeat() function when building your dataset.
3/3 [==============================] - 2s 684ms/step - loss: 9.8459e-05 - accuracy: 1.0000 - val_loss: 0.6551 - val_accuracy: 0.9032
Epoch 37/40
3/3 [==============================] - ETA: 0s - loss: 8.7555e-05 - accuracy: 1.0000WARNING:tensorflow:Your input ran out of data; interrupting training. Make sure that your dataset or generator can generate at least `steps_per_epoch * epochs` batches (in this case, 3 batches). You may need to use the repeat() function when building your dataset.
3/3 [==============================] - 2s 681ms/step - loss: 8.7555e-05 - accuracy: 1.0000 - val_loss: 0.6705 - val_accuracy: 0.9032
Epoch 38/40
3/3 [==============================] - ETA: 0s - loss: 8.4597e-05 - accuracy: 1.0000WARNING:tensorflow:Your input ran out of data; interrupting training. Make sure that your dataset or generator can generate at least `steps_per_epoch * epochs` batches (in this case, 3 batches). You may need to use the repeat() function when building your dataset.
3/3 [==============================] - 2s 662ms/step - loss: 8.4597e-05 - accuracy: 1.0000 - val_loss: 0.6844 - val_accuracy: 0.9032
Epoch 39/40
3/3 [==============================] - ETA: 0s - loss: 7.8716e-05 - accuracy: 1.0000WARNING:tensorflow:Your input ran out of data; interrupting training. Make sure that your dataset or generator can generate at least `steps_per_epoch * epochs` batches (in this case, 3 batches). You may need to use the repeat() function when building your dataset.
3/3 [==============================] - 2s 652ms/step - loss: 7.8716e-05 - accuracy: 1.0000 - val_loss: 0.6922 - val_accuracy: 0.9032
Epoch 40/40
3/3 [==============================] - ETA: 0s - loss: 7.5106e-05 - accuracy: 1.0000WARNING:tensorflow:Your input ran out of data; interrupting training. Make sure that your dataset or generator can generate at least `steps_per_epoch * epochs` batches (in this case, 3 batches). You may need to use the repeat() function when building your dataset.
3/3 [==============================] - 2s 698ms/step - loss: 7.5106e-05 - accuracy: 1.0000 - val_loss: 0.6944 - val_accuracy: 0.9032
predicted Lu. Confidence 100.00%
*****WRONG***** Choke expected

￼
predicted Lu. Confidence 100.00%
*****WRONG***** Choke expected

￼
predicted Lu. Confidence 100.00%
*****WRONG***** Choke expected

￼
predicted Choke. Confidence 100.00%

￼
predicted Choke. Confidence 100.00%

￼
predicted Lu. Confidence 100.00%

￼
predicted Lu. Confidence 100.00%

￼
predicted Lu. Confidence 100.00%

￼
predicted Choke. Confidence 100.00%
*****WRONG***** Lu expected

￼
predicted Lu. Confidence 100.00%

￼
predicted Lu. Confidence 100.00%

￼
predicted Lu. Confidence 100.00%

￼
predicted Lu. Confidence 100.00%

￼
predicted Lu. Confidence 100.00%

￼
predicted Lu. Confidence 100.00%

￼
predicted Lu. Confidence 100.00%

￼
predicted Lu. Confidence 100.00%

￼
predicted Choke. Confidence 100.00%
*****WRONG***** Noon expected

￼
predicted Noon. Confidence 100.00%

￼
predicted Noon. Confidence 100.00%

￼
predicted Noon. Confidence 100.00%

￼
predicted Noon. Confidence 100.00%

￼
predicted Choke. Confidence 100.00%
*****WRONG***** Noon expected

￼
wrong predictions = 6, correct predictions = 17

In [2]: 

