Contribution guidelines
=======================

Here we describe the workflow for contributing to the ``pyampute`` package.


Reporting bugs
##############

We use `GitHub issues`_ to track all bugs and feature requests; feel free to open an issue if you have found a bug or wish to see a feature implemented.

It is recommended to check that your issue complies with the following rules before submitting:

1. Verify that your issue is currently not being addressed by other issues or pull requests.

2. Please ensure all code snippets and error messages are formatted in appropriate code blocks.

3. Please include your operating system type and version number, as well as your Python, pyampute, scikit-learn and numpy versions.

.. _`GitHub issues`: https://github.com/RianneSchouten/pyampute/issues

Contributions
#############

All contributions to ``pyampute`` should be linked to `GitHub issues`_. If you encountered a particular part of the documentation or code that you want to improve, but there is no related open issue yet, please feel free to open one first. That is important it allows experienced contributors to give you feedback or pointers. Furthermore, to let everyone know that you are working on an issue, please leave a comment. 

.. _`GitHub issues`: https://github.com/RianneSchouten/pyampute/issues

Pull requests
#############

All contributions to ``pyampute`` should be done through `pull requests`_. To make a PR to ``pyampute``, you can do the following:

1. Fork the ``pyampute`` repository.

2. Create a feature branch to hold your development changes.

3. Develop your feature on the feature branch and add and commit changes.

4. Finally, follow `these instructions`_ to make a PR to the ``pyampute`` repository. That will send an e-mail to the owners. 

.. _`these instructions`: https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/creating-a-pull-request-from-a-fork
.. _`pull requests`: https://github.com/RianneSchouten/pyampute/pulls

Pull requests checklist
#######################

We recommend that your contribution complies with the following:

* Follow the `pep8`_ style guide, with the following exceptions; the max line length is 100 characters instead of 80, and when creating a multi-line expression with binary operators, break before the operator.

* Use the pull request title to describe the issue and mention the issue number in the pull request description. This will make sure a link back to the original issue is created.

* An incomplete contribution -- where you expect to do more work before receiving a full review -- should be submitted as a draft. These may be useful too; indicate you are working on something to avoid duplicated work, request broad review of functionality and/or seek collaborators. Drafts often benefit from the inclusion of a task list in the PR description.

* Add unittests and possibly an example for any new functionality being introduced.

* Documentation and high-coverage tests are necessary for enhancements to be accepted. Bug-fixes or new features should be provided with non-regression tests. These tests verify the correct behavior of the fix or feature. In this manner, further modifications are granted to be consistent with the desired behavior.

.. _`pep8`: https://www.python.org/dev/peps/pep-0008/