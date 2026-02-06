# Main Quest 1

[← Back to Main README](README.md) | [Next: Main Quest 2 →](main-quest-2.md)


## SMART Objectives


* Create a simple game using the MonoGame framework that compiles and runs successfully.
* Implement a screen management system using an enumerated type to switch between at several different game states (flash screen, title screen, credits screen, game screen, pause screen, gameover screen).
* Explain and demonstrate the role of the MonoGame Content Builder, showing how it compiles assets into .xnb files.
* Inherit from the Game class and correctly implement a class that extends its functionality.
* Override core game loop methods (Initialise, LoadContent, Update and Draw) and explain when each is called during execution.
* Process keyboard input to control a visible change in the game  (changing the screen colour).
* Render a coloured rectangle on the screen using MonoGame’s drawing methods.
* Display text on the screen and verify correct rendering in the game window.
* Position both a coloured rectangle and text on screen using screen coordinate values to demonstrate control over placement.
* Load a texture using the MonoGame Content Pipeline, and display it at a specified position on the screen.
* Process mouse input to detect clicks on a rectangle and trigger a visible response
* Record your reflections and submit your code to your GitHub repository.

## Completed Tasks

Currently I have implemented an enumerated type switching to different game states like flash screen, title screen, made a timer, processed mouse input which causes for the rectangle to dissapear and rendered a texture on the screen by using the monoGame content pipeline furthermore, implemented a high score system and made it all link up when you press the rectangle to go there and then to the title screen. 

## Reflection

<img width="802" height="519" alt="Screenshot 2025-09-30 083431" src="https://github.com/user-attachments/assets/40ca1c82-e2df-4a73-bf21-a1dc6d471039" />

This is the main thing you see first (flash screen) so you press the logo and depending at what time you do it at it gives you a score. It took me awhile to implement the code for it since it was a bit odd to use the different flash screens but I think I got used to it I also had some issues with the game screen not displaying correctly but I then realised that I have forgotten to add the rectangle and timer variables in the draw method for game screen. I also realised later on that it was supposed to be one rectangle with a border but I only realised after the fact that I did most of the work including the extra challenge for it. I think I will definetly need to look at it again to implement this. 

<img width="804" height="527" alt="image" src="https://github.com/user-attachments/assets/05f48754-981e-4de7-af43-5a5798278005" />

Credits screen is quite similar to the one in the labs, text shadow and all. Although, I did forget a few times to build the sprite font in the mono game loader thing but it does display your score in a linear fashion untill you hit 10s and if it's more than that it will give you an automatic game over with no score. I had a few issues with the timer not resetting and I am not exactly sure how I got it fixed. I have also implemented the extra challenge that being if you get a high score you get a congratulatory message that took a while to work mainly because i had to re-remember how to load files and stuff. 

<img width="804" height="527" alt="image" src="https://github.com/user-attachments/assets/8e3a8329-d651-40f3-a30a-fc342bde3bb2" />

This is the title screen but I have managed to fix it via messing around with the order list of the different game screens. it also displays the high scores from the high score.txt file it was originally in the game overscreen but I reread the extra challenge and I realised I have done it wrong so now it's in the correct position (I also realised that I've should've done copy if newer on the txt file it would have saved a lot of time). 
Final thoughts about the intro of monogame is that it's fairly interesting even if it was a bit hard to grasp in the begining like in some ways it does reminds me of graphics programing (probably because we are using openTK for the drawing of graphics) and how that draws graphics although instead of being done in seperate classes and inherited through them, monogame does it through the enum variables that being titleScreen, GameScreen etc. and you have that done through the update method. I haven't used monogame before but i have used game engines such as unity and it both feels the same and it doesn't. Monogame feels oddly enough like a more barebones version of unity or godot. 

## Beyond the Lab (Optional)


**Navigation:**
- [Main README](README.md)
- [Main Quest 2](main-quest-2.md)

