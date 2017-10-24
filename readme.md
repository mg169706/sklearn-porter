
# sklearn-porter

[![Build Status](https://img.shields.io/travis/nok/sklearn-porter/master.svg)](https://travis-ci.org/nok/sklearn-porter)
[![PyPI](https://img.shields.io/pypi/v/sklearn-porter.svg)](https://pypi.python.org/pypi/sklearn-porter)
[![PyPI](https://img.shields.io/pypi/pyversions/sklearn-porter.svg)](https://pypi.python.org/pypi/sklearn-porter)
[![GitHub license](https://img.shields.io/pypi/l/sklearn-porter.svg)](https://raw.githubusercontent.com/nok/sklearn-porter/master/license.txt)
[![Join the chat at https://gitter.im/nok/sklearn-porter](https://badges.gitter.im/nok/sklearn-porter.svg)](https://gitter.im/nok/sklearn-porter?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge)

Transpile trained [scikit-learn](https://github.com/scikit-learn/scikit-learn) estimators to C, Java, JavaScript and others.<br>It's recommended for limited embedded systems and critical applications where performance matters most.


## Machine learning algorithms

<table>
    <tbody>
        <tr>
            <td align="center" width="40%"><strong>Algorithm</strong></td>
            <td align="center" colspan="6" width="60%"><strong>Programming language</strong></td>
        </tr>
        <tr>
            <td align="left" width="40%">Classification</td>
            <td align="center" width="10%">C</td>
            <td align="center" width="10%">Java *</td>
            <td align="center" width="10%">JavaScript</td>
            <td align="center" width="10%">Go</td>
            <td align="center" width="10%">PHP</td>
            <td align="center" width="10%">Ruby</td>
        </tr>
        <tr>
            <td><a href="http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html">sklearn.svm.SVC</a></td>
            <td align="center"><a href="examples/estimator/classifier/SVC/c/basics.ipynb">✓</a></td>
            <td align="center"><a href="examples/estimator/classifier/SVC/java/basics.ipynb">✓</a></td>
            <td align="center"><a href="examples/estimator/classifier/SVC/js/basics.ipynb">✓</a></td>
            <td align="center"></td>
            <td align="center"><a href="examples/estimator/classifier/SVC/php/basics.ipynb">✓</a></td>
            <td align="center"><a href="examples/estimator/classifier/SVC/ruby/basics.ipynb">✓</a></td>
        </tr>
        <tr>
            <td><a href="http://scikit-learn.org/stable/modules/generated/sklearn.svm.NuSVC.html">sklearn.svm.NuSVC</a></td>
            <td align="center"><a href="examples/estimator/classifier/NuSVC/c/basics.ipynb">✓</a></td>
            <td align="center"><a href="examples/estimator/classifier/NuSVC/java/basics.ipynb">✓</a></td>
            <td align="center"><a href="examples/estimator/classifier/NuSVC/js/basics.ipynb">✓</a></td>
            <td align="center"></td>
            <td align="center"><a href="examples/estimator/classifier/NuSVC/php/basics.ipynb">✓</a></td>
            <td align="center"><a href="examples/estimator/classifier/NuSVC/ruby/basics.ipynb">✓</a></td>
        </tr>
        <tr>
            <td><a href="http://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html">sklearn.svm.LinearSVC</a></td>
            <td align="center"><a href="examples/estimator/classifier/LinearSVC/c/basics.ipynb">✓</a></td>
            <td align="center"><a href="examples/estimator/classifier/LinearSVC/java/basics.ipynb">✓</a></td>
            <td align="center"><a href="examples/estimator/classifier/LinearSVC/js/basics.ipynb">✓</a></td>
            <td align="center"><a href="examples/estimator/classifier/LinearSVC/go/basics.ipynb">✓</a></td>
            <td align="center"><a href="examples/estimator/classifier/LinearSVC/php/basics.ipynb">✓</a></td>
            <td align="center"><a href="examples/estimator/classifier/LinearSVC/ruby/basics.ipynb">✓</a></td>
        </tr>
        <tr>
            <td><a href="http://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html">sklearn.tree.DecisionTreeClassifier</a></td>
            <td align="center"><a href="examples/estimator/classifier/DecisionTreeClassifier/c/basics.ipynb">✓</a></td>
            <td align="center"><a href="examples/estimator/classifier/DecisionTreeClassifier/java/basics.ipynb">✓</a></td>
            <td align="center"><a href="examples/estimator/classifier/DecisionTreeClassifier/js/basics.ipynb">✓</a></td>
            <td align="center">✓</td>
            <td align="center"><a href="examples/estimator/classifier/DecisionTreeClassifier/php/basics.ipynb">✓</a></td>
            <td align="center">✓</td>
        </tr>
        <tr>
            <td><a href="http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html">sklearn.ensemble.RandomForestClassifier</a></td>
            <td align="center"><a href="examples/estimator/classifier/RandomForestClassifier/c/basics.ipynb">✓</a></td>
            <td align="center"><a href="examples/estimator/classifier/RandomForestClassifier/java/basics.ipynb">✓</a></td>
            <td align="center"><a href="examples/estimator/classifier/RandomForestClassifier/js/basics.ipynb">✓</a></td>
            <td align="center"></td>
            <td align="center">✓</td>
            <td align="center">✓</td>
        </tr>
        <tr>
            <td><a href="http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.ExtraTreesClassifier.html">sklearn.ensemble.ExtraTreesClassifier</a></td>
            <td align="center"><a href="examples/estimator/classifier/ExtraTreesClassifier/c/basics.ipynb">✓</a></td>
            <td align="center"><a href="examples/estimator/classifier/ExtraTreesClassifier/java/basics.ipynb">✓</a></td>
            <td align="center"><a href="examples/estimator/classifier/ExtraTreesClassifier/js/basics.ipynb">✓</a></td>
            <td align="center"></td>
            <td align="center">✓</td>
            <td align="center">✓</td>
        </tr>
        <tr>
            <td><a href="http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html">sklearn.ensemble.AdaBoostClassifier</a></td>
            <td align="center"><a href="examples/estimator/classifier/AdaBoostClassifier/c/basics.ipynb">✓</a></td>
            <td align="center"><a href="examples/estimator/classifier/AdaBoostClassifier/java/basics.ipynb">✓</a></td>
            <td align="center"><a href="examples/estimator/classifier/AdaBoostClassifier/js/basics.ipynb">✓</a></td>
            <td align="center"></td>
            <td align="center"></td>
            <td align="center"></td>
        </tr>
        <tr>
            <td><a href="http://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html">sklearn.neighbors.KNeighborsClassifier</a></td>
            <td align="center"></td>
            <td align="center"><a href="examples/estimator/classifier/KNeighborsClassifier/java/basics.ipynb">✓</a></td>
            <td align="center"><a href="examples/estimator/classifier/KNeighborsClassifier/js/basics.ipynb">✓</a></td>
            <td align="center"></td>
            <td align="center"></td>
            <td align="center"></td>
        </tr>
        <tr>
            <td><a href="http://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html">sklearn.neural_network.MLPClassifier</a></td>
            <td align="center"></td>
            <td align="center"><a href="examples/estimator/classifier/MLPClassifier/java/basics.ipynb">✓</a></td>
            <td align="center"><a href="examples/estimator/classifier/MLPClassifier/js/basics.ipynb">✓ ᵂᵂ</a></td>
            <td align="center"></td>
            <td align="center"></td>
            <td align="center"></td>
        </tr>
        <tr>
            <td><a href="http://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.GaussianNB.html#sklearn.naive_bayes.GaussianNB">sklearn.naive_bayes.GaussianNB</a></td>
            <td align="center"></td>
            <td align="center"><a href="examples/estimator/classifier/GaussianNB/java/basics.ipynb">✓</a></td>
            <td align="center"><a href="examples/estimator/classifier/GaussianNB/js/basics.ipynb">✓</a></td>
            <td align="center"></td>
            <td align="center"></td>
            <td align="center"></td>
        </tr>
        <tr>
            <td><a href="http://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.BernoulliNB.html#sklearn.naive_bayes.BernoulliNB">sklearn.naive_bayes.BernoulliNB</a></td>
            <td align="center"></td>
            <td align="center"><a href="examples/estimator/classifier/BernoulliNB/java/basics.ipynb">✓</a></td>
            <td align="center"><a href="examples/estimator/classifier/BernoulliNB/js/basics.ipynb">✓</a></td>
            <td align="center"></td>
            <td align="center"></td>
            <td align="center"></td>
        </tr>
        <tr>
            <td align="left" width="40%">Regression</td>
            <td colspan="6" width="10%"></td>
        </tr>
        <tr>
            <td><a href="http://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPRegressor.html">sklearn.neural_network.MLPRegressor</a></td>
            <td align="center"></td>
            <td align="center"></td>
            <td align="center"><a href="examples/estimator/regressor/MLPRegressor/js/basics.ipynb">✓ ᵂᵂ</a></td>
            <td align="center"></td>
            <td align="center"></td>
            <td align="center"></td>
        </tr>
    </tbody>
</table>

✓ = is full-featured, ○ = has minor exceptions, ᵂᵂ = supports [Web Workers](https://www.w3.org/TR/workers/), * = default language

## Installation

```bash
pip install sklearn-porter
```

If you want the [latest changes](changelog.md#unreleased), you can install the module from the master (development) branch:

```bash
pip uninstall -y sklearn-porter
pip install --no-cache-dir https://github.com/nok/sklearn-porter/zipball/master
```

## Minimum requirements

```
- python>=2.7.3
- scikit-learn>=0.14.1
```

If you want to transpile a [multilayer perceptron](http://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html), you have to upgrade the scikit-learn package:

```
- python>=2.7.3
- scikit-learn>=0.18.0
```


## Usage

### Export

The following example shows how you can port a [decision tree model](http://scikit-learn.org/stable/modules/tree.html#classification) to Java:

```python
from sklearn.datasets import load_iris
from sklearn.tree import tree
from sklearn_porter import Porter

# Load data and train the classifier:
samples = load_iris()
X, y = samples.data, samples.target
clf = tree.DecisionTreeClassifier()
clf.fit(X, y)

# Export:
porter = Porter(clf, language='java')
output = porter.export()
print(output)
```

The exported [result](examples/estimator/classifier/DecisionTreeClassifier/java/basics.py#L18-L98) matches the [official human-readable version](http://scikit-learn.org/stable/_images/iris.svg) of the decision tree.

### Prediction

Run the prediction(s) in the target programming language directly:

```python
# ...
porter = Porter(clf, language='java')

# Prediction(s):
Y_preds = porter.predict(X)
y_pred = porter.predict(X[0])
y_pred = porter.predict([1., 2., 3., 4.])
```

### Accuracy

Always compute the accuracy between the original and the ported estimator:

```python
# ...
porter = Porter(clf, language='java')

# Accuracy:
accuracy = porter.predict_test(X)
print(accuracy) # 1.0
```


### Command-line interface

This example shows how you can port a model from the command line. First of all you have to store the model to the [pickle format](http://scikit-learn.org/stable/modules/model_persistence.html#persistence-example):

```python
# ...

# Extract estimator:
joblib.dump(clf, 'estimator.pkl')
```

After that the model can be transpiled by using the following command:

```
python -m sklearn_porter --input <PICKLE_FILE> [--output <DEST_DIR>] [--class_name <NAME>] [--method_name <NAME>] [--pipe] [--c] [--java] [--js] [--go] [--php] [--ruby]
python -m sklearn_porter -i <PICKLE_FILE> [-o <DEST_DIR>] [--class_name <NAME>] [--method_name <NAME>] [-p] [--c] [--java] [--js] [--go] [--php] [--ruby]
```

For instance the following command transpiles the estimator to the target programming language JavaScript:

```
python -m sklearn_porter -i estimator.pkl --js
```

The target programming language is changeable on the fly: 

```
python -m sklearn_porter -i estimator.pkl --c
python -m sklearn_porter -i estimator.pkl --go
python -m sklearn_porter -i estimator.pkl --php
python -m sklearn_porter -i estimator.pkl --java
python -m sklearn_porter -i estimator.pkl --ruby
```

The transpiled estimator is useable for further processing by using the `--pipe` parameter:

```
python -m sklearn_porter -i estimator.pkl --js --pipe > estimator.js
```

For instance the generated JavaScript code can be minified by using [UglifyJS](https://github.com/mishoo/UglifyJS2):

```
python -m sklearn_porter -i estimator.pkl --js --pipe | uglifyjs --compress -o estimator.min.js 
```

Further information will be shown by using the `--help` parameter:

```
python -m sklearn_porter --help
python -m sklearn_porter -h
```


## Development

### Environment

Install the required [environment modules](environment.yml) by executing the script [environment.sh](recipes/environment.sh):

```bash
bash ./recipes/environment.sh
```

```bash
conda env create -c conda-forge -n sklearn-porter python=2 -f environment.yml
source activate sklearn-porter
```

The following compilers or intepreters are required to cover all tests:

- [GCC](https://gcc.gnu.org) (`>=4.2`)
- [Java](https://java.com) (`>=1.6`)
- [PHP](http://www.php.net/) (`>=7`)
- [Ruby](https://www.ruby-lang.org) (`>=2.4.1`)
- [Go](https://golang.org/) (`>=1.7.4`)
- [Node.js](https://nodejs.org) (`>=6`)

### Testing

The tests cover module functions as well as matching predictions of transpiled models. Run [all tests](tests) by executing the script [test.sh](recipes/test.sh):

```bash
bash ./recipes/test.sh
```

```bash
python -m unittest discover -vp '*Test.py'
```

The test files have a specific pattern: `'[Algorithm][Language]Test.py'`:

```bash
python -m unittest discover -vp 'RandomForest*Test.py'
python -m unittest discover -vp '*JavaTest.py'
```

While you are developing new features or fixes, you can reduce the test duration by setting the number of model tests:

```bash
N_RANDOM_FEATURE_SETS=15 N_EXISTING_FEATURE_SETS=30 python -m unittest discover -vp '*Test.py'
```


### Quality

It's highly recommended to ensure the code quality. For that I use [Pylint](https://github.com/PyCQA/pylint/), which you can run by executing the script [lint.sh](recipes/lint.sh): 

```bash
bash ./recipes/lint.sh
```

```bash
find ./sklearn_porter -name '*.py' -exec pylint {} \;
```


## License

The module is Open Source Software released under the [MIT](license.txt) license.


## Questions?

Don't be shy and feel free to contact me on [Twitter](https://twitter.com/darius_morawiec) or [Gitter](https://gitter.im/nok/sklearn-porter).
