# Explore_mosaic
Implementation of the data exploration Python application for multi-sensor correlation analysis and image mosaic display.

![ezgif com-gif-maker](https://user-images.githubusercontent.com/17483097/149989862-9b6159aa-95ce-44e5-acec-6c43da59676a.gif)


## Requirements
* Python >= 3.8.0
* Numpy == 1.19.5
* OpenCV == 4.5.1
* PyPI - utm == 0.7.0
* Matplotlib == 3.3.4
* Scikit-learn == 0.23.2

## Prerequisites
* Download the [metadata](https://drive.google.com/file/d/1fULGE1w_DcB7MA02JcIy0g7kYwiC5z08/view?usp=sharing) and unzip it to ```/data/scallops/```
* Download the MAT files and sonar waterfall images to ```/data/scallops/sonar/```: [20170817_IM](https://drive.google.com/file/d/1peYBaSDPaXbjzY8LGifPp7zFOTS0LuDT/view?usp=sharing), [20170824_VIMS](https://drive.google.com/file/d/1f2f9vOWJVlyf9FIP1pSm-dsCn-AqScX5/view?usp=sharing)
* Download the SuperGlue registration numpy files to ```/data/scallops/superglue_output/```: [20170817_IM](https://drive.google.com/file/d/1f2f9vOWJVlyf9FIP1pSm-dsCn-AqScX5/view?usp=sharing), [20170824_VIMS](https://drive.google.com/file/d/1WAKyMJhuBSrxGlUDwYp3ASA6bkOZzhVl/view?usp=sharing)

## Usage
The script takes two arguments: index of the camera image to show, and the mission name
```
# Running directly from the repository
python3 explore_mosaic.py 3000 20170817_IM
```

Keyboard controls: 

```Space```   - Flip order of camera and sonar in overlay

```0```       - Reset scale and origin of image mosaic

```w/a/s/d```	-	Translate image mosaic


```t/f/h/f```	- Big translation of image mosaic

```k/l```		  - Prev/next leg neighbor

```	5-9```		- Preselected zoom levels

```	-/=```		- Scale image mosaic

```'```	 		  - Next neighbor camera image

```;```			  - Prev neighbor camera image

```x```		    - Next sonar

```z```			  - Prev sonar

```]/[```			- Next/prev absolute image

```p/o```			- +10/-10 absolute image 

```i/u```			- +100/-100 absolute image

```m```			- Toggle display of frame indices and altitudes

```n/b```		- +/- sonar alignment factor

```1```		- Display mode normal

```2```		- Display mode mask

```q```		- Quit
```
