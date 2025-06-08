# BCI-imagery-paradigm
This repo contain the python code to record EEG sessions while recording markers 
## 1. Clone the Repository
First, clone this GitHub repository to your local machine:
'''
git clone <repository_url>
cd <repository_name>
'''
Replace <repository_url> with the actual URL of this repository and <repository_name> with the name of the cloned directory.

## 2. Create a Python Virtual Environment
It is highly recommended to use a virtual environment to manage project dependencies. This isolates the project's packages from your global Python installation, preventing conflicts.

Navigate to your project's root directory (where requirements.txt is located) in your terminal or command prompt and run:
'''
conda create -n bci_env python=3.10.10
'''

## 3. Activate the Virtual Environment
Before installing dependencies or running project scripts, you must activate the virtual environment.
'''
conda activate bci_env

You will know the virtual environment is active when you see (bci_env) (or your chosen environment name) at the beginning of your terminal prompt.

## 4. Install Dependencies
With the virtual environment activated, you can now install all the required Python packages listed in requirements.txt:
'''
pip install -r requirements.txt
'''
This command reads the requirements.txt file and installs all specified packages and their exact versions into your bci_env environment.

## 5. Deactivate the Virtual Environment
When you are finished working on the project, you can deactivate the virtual environment:
'''
conda deactivate
'''
Your terminal prompt will return to its normal state, indicating that the virtual environment is no longer active. You can reactivate it anytime you return to work on the project.

