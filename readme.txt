Since h5 keras model and image dataset is too big to upload to CVUT thesis system. Please.

!!!!!! DOWNLOAD WHOLE PROJECT !!!!!!!!!!

from here: https://drive.google.com/drive/folders/1bxKhDII7J9sMV51ajXjBq4nVTtlBRnDt?usp=sharing

!!!!!! DOWNLOAD WHOLE PROJECT !!!!!!!!!!

with project downloaded we can continue

0.0) dataset
	to edit and browse labeled dataset one needs to instal labelme:
	
		pip install labelme
	
	then launch it 
	
		labelme
		
	select folder dataset and stare in ave how precisely the plastrons were labeled.

1.1) detection folder

detection folder contains all python source codes, it should be installed by 

	pip install .

since some pyx files needs to be cythonized before running also it collects necesray requirements
installation of package cynetworkx (cythonized networkx, same functions but shuld be faster) is bit lenghty and it sucess depends on some OS requirements (gcc and libc ... depends on os).

or its possible to build docker image and work inside of it (then you will have to connect the images folders as volumes)

	docker build . -t gads
	
Name of the image is optional but i will refer to it as gads, its possible to run it like this.

	docker run -p 5000:5000 gads

Entry point of the image launches flask server on port 5000. To test its functionality one might want to use curl 
like this:
 
	curl -X POST -F "file=@../dataset/Tg196b.jpg" 127.0.0.1:5000
	 
response is json with validation message and base64 encoded image with marked detection results

1.2) detection/plastron folder

	To run detection and preprocesing of images (detection, rotation and croping) from thesis content run:
	
		python detection\plastron\plastron_detector.py  -i dataset -o detected_plastrons -m DETECT
		
	There is also short scritp to launch training but it requires user to divide the dataset folder into train and test folders. This was also launched only on google colab.

1.3) detection/junction

	To run detection of junctions launch:
	
		python detection\junction\JunctionDetector.py -i detected_plastrons -o detected_plastrons
		
	Whole junction detection algorythm is in file JunctionDetector.py
	
1.4) detection/gads
		
	The pinacle also so called cherry on the cake of the project cythonized optimized gads algorythm in file GADS.pyx
	There are no launchable parts except unit/integration/performance test in detection/gads/test/TestGraphBuilders.py
	
1.5) detection/evaluation

	To run evaluation launch from detection/evaluation:
	
		python detection\evaluation\evaluation_features.py
	
	this should populate the folder detection\evaluation\figs with top notch vizualizations and many other surprisingly good things...
	
1.6) detection/server
	
	flask app servig the detection of plastrons functionality
	
	
2.0) turtle-camera-android

	Project with mobile client There is APK in: 
	
		turtle-camera-android\app\release\app-release.apk
		
3.0) detected_plastrons

	Outputs of plastron detection step. Cropped images,  marked images to eyeball the results, and translated test labels for junctions.
	
4.0) detected_junctions

	Outputs of junction detection step. Images with marked seams, preprocesed binary images and json files with junction coordinates.
		
		
		
	
		

	
	





	