# MFV - mosaic the face in the video

### A mosaic processing program for all but a specific person by recognizing the face in the video.

#### Sample video
<img width="80%" src="https://user-images.githubusercontent.com/37572031/146848665-9d93c54b-293b-4e4a-8625-2b0b7d0cf17c.gif"/>

#### Folder description
- Faces: Photos of people who won't mosaic.
	- Possibility multiple people.
	- If you put pictures from various angles, the accuracy of excluding mosaic will be improved.
- Model: Learning model
	- haarcascade_frontface_alt.xml model
- Result : Program execution result video
	- If there's no folder, it's made on its own.
	- The result video of the mosaic is saved.
- Video: The video file to be mosaic
	- It includes a video clip to cover face with mosaic.

#### Project language
- Python

#### Reference site
- https://deep-eye.tistory.com/18#google_vignette
- https://ichi.pro/ko/paisseon-eulo-eolgul-insiggi-mandeulgi-140558761485120
- https://ponyozzang.tistory.com/598?category=800537
