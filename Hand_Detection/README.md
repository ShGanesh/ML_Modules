# Hand Detection  
[HandTrackingModule.py](./HandTrackingModule.py) is a module for hand-tracking that was made using [mediapipe](https://pypi.org/project/mediapipe/).  
Imma use this module in future projects.

## Usage  
The module contains a dummy `main()` inside it, but a [test script](./hand_test.py) was also made. You can copy a skeleton code from the test script to further customize the hand tracking according to your wishes.

## Input  
Test Script takes webcam input.

## Output:
Test script outputs:
- CV2 video output with hand landmarks highlighted and connected.
- Rough FPS rate on the upper left-hand corner.
- Real-time location of the tip of the thumb in each hand.

## Significant Constraints:  
Module is configured to only detect 2 hands. It can be changed tho.

## Media  
[Video 1](./MLH_INIT_Day1_Explore_ML.mp4): Shows a demo of our _(what "our"? its "my")_ working [test script](./hand_test.py).  
[Picture 1](./MLH_INIT_Day1_Explore_ML_1.png): Shows one image from output screen.  
[Picture 2](./MLH_INIT_Day1_Explore_ML_2.png): Shows one image from output screen.
[Picture 3](./MLH_INIT_Day1_Explore_ML_results.png): Shows results of the script in a linux terminal.

## Shoutouts:  
Shoutouts to [freeCodeCamp.org](https://www.youtube.com/channel/UC8butISFwT-Wl7EV0hUK0BQ) and [Murtaza Hassan](https://www.youtube.com/channel/UCYUjYU5FveRAscQ8V21w81A) for tutorials on mediapipe and ML.
