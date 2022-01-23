![build_status](https://github.com/paiforsyth/qml/actions/workflows/pants.yaml/badge.svg)

## Install pre-commit

```
pre-commit install
```

## Create virtual environment for use with IDEs (for example, Pycharm)

```
gen_env.sh
```

Activate it using
```
chmod +x ../ide_venv/bin/activate
../ide_venv/bin/activate
```

## List targets

```
./pants list ::  # All targets.
./pants list 'helloworld/**/*.py'  # Just targets containing Python code.
```

## Run linters and formatters

```
./pants lint ::
./pants fmt 'helloworld/**/*.py'
```

## Run MyPy

```
./pants typecheck ::
```

## Run tests

```
./pants test ::  # Run all tests in the repo.
./pants test helloworld/util:test  # Run all the tests in this target.
./pants test helloworld/util/lang_test.py  # Run just the tests in this file.
./pants test helloworld/util/lang_test.py -- -k test_language_translator  # Run just this one test.
```

## Create a PEX binary

```
./pants package helloworld/main.py
```

## Run a binary

```
./pants run helloworld/main.py
```

## Open a REPL

```
./pants repl helloworld/greet  # The REPL will have all relevant code and dependencies on its sys.path.
./pants repl --shell=ipython helloworld/greet
```

## Build a wheel / generate `setup.py`

This will build both a `.whl` bdist and a `.tar.gz` sdist.

```
./pants package helloworld/util:dist
```

We can also remove the `setup_py_commands` field from `helloworld/util/BUILD` to have Pants instead generate a
`setup.py` file, with all the relevant code in a chroot.

## Count lines of code

```
./pants count-loc '**/*'
```

## Update requirements using pur

```
./pants run qml/tools/run_pur.py
```

## Troubleshooting
1. An out of date constraints.txt file can cause obscure errors.  Try
```
build-support/generate_constraints.sh
```
