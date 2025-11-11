Step 4: Create the Backend Server (app.py)
📋 Instructions
Goal: To create the main Python file that will run your web server, load the AI model, and handle all the logic.

Make sure you are in your project directory (~/ai_assistant).

Your virtual environment must be active. If it's not, run source venv/bin/activate. You should see (venv) at the start of your terminal prompt.

Create a new file named app.py using a text editor (like nano or gedit).

If you're in the terminal, you can type nano app.py to create and edit it.

Copy all the code from the block below and paste it into your new app.py file.

Read the comments in the code. They explain what each part does.

Important: Note the N_GPU_LAYERS = 0 line. This is set for CPU-only operation. If you have a powerful NVIDIA GPU, you could change this later, but for now, 0 is the safest, most compatible option.

Save the file and exit the text editor. (In nano, you press Ctrl+O to save, then Enter, then Ctrl+X to exit).

This file is the "engine" of your entire application. It won't do anything just by existing, but we will run it in a later step.

💻 Code (app.py)
(Copy this entire block into your app.py file)
