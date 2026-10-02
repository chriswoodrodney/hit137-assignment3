SYNOPSIS OF THE SRAMBLED IMAGE PUZZLE

 Developed a desktop application that demonstrates our understanding of OOP,GUI development using Tkinter, and image processing using OpenCV.

# Requirements
opencv-python
numpy

# Player guide
1. Load thew image, choose the image size (3*3, 4*4, 5*5), choose timer to make it more interesting 
2.  Left click a tile, Select it (orange border)
3. Left click a second tile, Swap the two tiles, selection cleared 
4. Left click the selected tile again,Deselect 
5. Right click a tile, Rotate 90° clockwise 
6. Shift + left click a tile, Flip horizontally 

# Hint
Blue circle on one wrong tile and on its home position in the original, it's cleared by the next move. Player has 3 hints per image, then the button is disabled. 

# Solve
Undoes all transformations instantly and clears the moves and score, basically solves the puzzle when a player feels like they have failed or cannot continue

# Timer 
Found at the bottom right of the score panel, with no limit it counts up (`Time: 0:42`),with a limit it counts down (`Time left: 1:18`) and turns red in the last 10 seconds. It stops when the puzzle is solved.

# Time's up!
When the countdown reaches 0:00 a message is shown, the puzzle is locked (clicks, Hint and Solve are disabled) and you can load another image to try again. 

 # Supported grid sizes
 3 x 3
 4 x 4
 5 x 5

 # TEAM CONTRIBUTION GROUP 17
 Member                 |STUD ID | CONTRIBUTION                                      |PERCENTAGE/RATE
Chriswood RODNEY OKWIIRI| s408138| setup project configuration and core tile classes |50%
                                 | transformations and core board
                                 | image processor, scrambler and timer

SAFA ALAM               |        | puzzle game engine and game rules                 |50%
                                 | Tkinter intrface and application entry point 
                                 | readme.


# NOTE:
Commits were made by Chriswood because Safa was sick and asked for me, sent her work contribution ti Chriswood who committed and pushed the work on the Github. But contribution was divided because both group members contributed.                               