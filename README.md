## Dependencies

First, I suggest you create a new python environment. Execute this at the root of the project:

```
python3 -m venv .venv
source .venv/bin/activate
```

Then, install the dependencies:

```
pip3 install -r requirements.txt
```

## Running the scripts

You may then run the task1 and task2 scripts. They both require as an argument the path to the input directory, and the path where to store the results:

```
python3 task1.py train solution
python3 task2.py train solution
```

Or:

```
python3 task1.py evaluation/fake_test solution
python3 task2.py evaluation/fake_test solution
```
