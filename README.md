# python-project-template

This is a simple python project template for Visual studio code. 

Create and activate virtual environment:

   ```sh
   python -m venv .venv
   ```
   ```sh
   "./.venv/Scripts/activate"
   ```
   
   or
   
   ```sh
   conda create -n yourenvname python=x.x anaconda
   ```
   ```sh
   conda activate yourenvname
   ```
   
Clear git cached files and directories:

   ```sh
   git rm --cached -r .vscode 
   ```
   ```sh
   git rm --cached .env
   ```
    
Set path to project root directory in `.env`, e.g.:

   ```sh
   PYTHONPATH=C:\\Users\\janezla\\Documents\\python-project-template
   ```
   
Set python path in vscode workspace settings, e.g.:
   ```sh
   "python.pythonPath": "C:\\Users\\janezla\\Anaconda3\\envs\\yourenvname\\python"
   ```

