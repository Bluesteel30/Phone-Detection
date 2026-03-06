# Phone-Detection
## Why
At Westwood High School students are supposed to put their phones into the phone "cubbies." Despite this requirement, many students do not comply or forget, leading teachers to take valuable time that they could be teaching to cross refrence attendance and the phone "cubbies." As part of an AP CSA project we decided to see if we could mitigate this waste of time. 

## How
First we attached ArUco markers to the corners of the phone "cubby." Then using the OpenCV Library we were able to digitally flatten an image. This mitigrates the role that the angle of which the picture is taken plays in wraping the image. We then scaled the image to 1000x1500. By flattening and scalling the image to a known resolution, we are able to define boxes around each phone slot. Then using Roboflows API, we are able to use their Sam3 model to search for phones. Calling this api returns pixel coordinates of the center of each phone. Using this, we can check if the center of a phone is within a certain box. Each box is linked through a dictionary to a student. The dictionaries can be changed to repersent the different blocks of the day. If a name is assigned to a box, and no phone is detected within it, then the student's name is outputted to the terminal. For this project we were tasked with using Generative AI. We mainly used Gemini, as WHS uses Google Workspace. 

## Demo
We can take a warped image like

https://github.com/user-attachments/assets/60cc62b7-a8bd-4dfb-8132-24829c71a4ac

and through merge.py

https://github.com/user-attachments/assets/674f9a0c-1596-445f-9881-631763bc01c1


