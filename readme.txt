
Repo restarted Feb 14 2025 to avoid gitlog issue

A. # deactivate and delete venv before creating a new venv

1. To create a virtual environment 
    py/python -m venv .venv
    # .venv\Scripts\activate
    # pip install -r requirements.txt
    # activate venv and run program py app.py

2. activate venv a (venv) shold appear before the file path in terminal ie "(venv) PS C:\Users\ihiggins\Desktop\cache_hydro_data>"
.\venv\Scripts\activate 

3. install required packagews
    pip install -r requirements.txt
    I often need to update pip? -m pip install --upgrade pip


B. to deactivate venv
    "deactivate"

C. To make requirements.txt 
1. if requirements.txt exists delete it first (you may be able to update?) * updating may be prefered
2. run py -m pip freeze > requirements.txt
3. I have had iuuses with having extra packages show up in my requirements.txt so it may be best to just modify the existing file

 Note: you may have to use python, or python3 instead of py

 D. Install  packages
 [py/python] is the python interpreter
 [-m] module: allows you to run python modules as a Scripts
 [pip/conda] package manager
 [py] [-m] [pip] install [package]


source venv/bin/activate # uses the virtualenv

# a conda environment was created in cache_hydro_data and should be used
py -m conda install <package>

# compile
# use pyinstaller but pyinstaller does not save to path to get location run pip show pyinstaller
 King County\Desktop\cache_hydro_data> python "c:\users\ihiggins\appdata\local\packages\pythonsoftwarefoundation.python.3.9_qbz5n2kfra8p0\localcache\local-packages\python39\site-packages\pyinstaller" cache_gdata.py

python "c:\users\ihiggins\appdata\local\packages\pythonsoftwarefoundation.python.3.9_qbz5n2kfra8p0\localcache\local-packages\python39\site-packages\pyinstaller" dash_test