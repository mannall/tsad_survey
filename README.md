# Time Series Anomaly Detection Survey

### Linux installation [[Source](https://packaging.python.org/en/latest/guides/installing-using-pip-and-virtual-environments/)]
Download repository
```sh
git clone git@github.com:mannall/tsad_survey.git
cd ./tsad_survey
```

Setup Python virtual environment named ".venv", activate virtual environment ".venv" and verufy installation
```sh
python3 -m venv .venv
source .venv/bin/activate
which python 
# ... /home/user/project_name/venv/bin/python ...
```

Install dependencies
```sh
pip install -r requirements.txt
```

Test anomaly detection algorithms on example data
```sh
python3 test.py
```

When finished deactivate virtual environment
```sh
deactivate
```