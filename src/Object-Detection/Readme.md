This is the code to train a prototype YOLOv11n and test the model after training.
Training code is using C++ to request Python commands to do the training, validate it and in case of failure during training resume training based on the last check point (save_period).
This is a prototype and many of the variables were hard coded.

A demo prototype was created to test the accuracy of the taining by asking the user to provide a test video.
The demo shows the frames from the video with annotations showing the detected objects, the class and confidence index, and the frame process ratio of the detection.
A log is kept for each object detected in each frame, and a location in the image.
A samll video is produced during the demo as part of documented the detection outcome. this is a hard coded video name.
The user can stop the analysis by pressing ESC, "Q", or "q".

A makefile is also included so the source can be compile and then test it.
