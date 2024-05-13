# Casper DPM

This is the official implementation of Casper DPM: Cascaded Perceptual Dynamic Projection Mapping onto Hands

## Installation
### Requirements
- OS: Windows 10 or above
- Visual Studio 19 or above
- CMAKE 3.20 or above
- [OpenCV](https://opencv.org/releases/) 4.6 or above
- A python interpreter 3.9 or above (with dev tools, which is usually included in windows versions), with numpy and mediapipe installed:
  `pip install numpy mediapipe`
  
Optional:
- [Pylon SDK](https://www.baslerweb.com/en/software/pylon/sdk/) - for basler camera control
- [Gemini Ultra Leap](https://www.ultraleap.com/tracking/gemini-hand-tracking-platform/) - for leap motion controller API
### How to install
1) Clone the repo (let's assume you cloned into "C:\src\casper"):

  `git clone https://github.com/yoterel/casper.git`
   
2) Change dir:

  `cd C:\src\casper`

3) setup paths in CMakeLists.txt
   This step is manual.
   - set the variable Python_EXECUTABLE to the python executable you will use, for now only a system python was tested and not virtual environments.
   - set the variable OpenCV_DIR to the Opencv install path. Or, place the openCV install path under casper/third-party/opencv.
   - other paths should be detected automatically using environment variables, check error msgs in the next step and replace any missing path.
     
3) configure cmake (this assumes visual studio 2019 is your generator):

  `cmake -SC:/src/casper =BC:/src/casper/build -G "Visual Studio 16 2019" -T host=x64 -A x64`


4) build project:

  `cmake --build C:/src/casper/build --config Release --target ALL_BUILD -j 10 --`

## Usage

- to run casper in sandbox mode (this will allow you to perform any task that was described in the paper):

  `casper.exe`

Right clicking with the mouse will open a gui with all configurations

- to run a discrete simulation:
  `casper.exe --mode simulation`

To start you can open the menu and check "Debug Playback" under playback settings.

- if you don't have a projector or a camera, but still wish to check out casper, you can run with a simulated camera and projector:
  
  `casper.exe --simcam --simproj`
