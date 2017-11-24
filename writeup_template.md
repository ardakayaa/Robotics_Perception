# WRITEUP

# Capturing features and Training Models for Object Detection
- There are eight types of objects: biscuits, soap, soap2, book, glue, sticky notes, snacks, and eraser. 
- To train the model, 
[**THIS SCRIPT**](https://github.com/ardakayaa/Robotics_Perception/blob/master/pr2_robot/scripts/capture_features.py)
spawns each object in **10 random orientations** and computes features based on the point clouds 
resulting from each of the random orientations.
- The features are normalized histograms of the color and normal 3d vectors for each point in the point cloud
captured by the virtual RGBD camera. The color is expressed in **HSV** format, because they capture the true color better,
regardless of lighting conditions. The normal vectors for each point capture the shape of object. You can inspect the 
[**SCRIPT HERE**](https://github.com/ardakayaa/Robotics_Perception/blob/master/pr2_robot/scripts/features.py)
- I used 
[**THIS SCRIPT**](https://github.com/ardakayaa/Robotics_Perception/blob/master/pr2_robot/scripts/train_svm.py)
to train the classifier which is a support vector machine classifier. 
- Check out the resulting confusion matrix below. 
![training result](https://github.com/ardakayaa/Robotics_Perception/blob/master/img/Screenshot%20from%202017-11-24%2016-56-12.png)

