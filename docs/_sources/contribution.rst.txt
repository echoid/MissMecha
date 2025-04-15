Contribution Guidelines
=======================

We warmly welcome contributions to the ``MissMecha`` project!  
Whether it's a bug fix, a new feature, improved documentation, or even a thoughtful suggestion — every contribution helps improve the package.

This guide describes the recommended workflow when contributing to MissMecha.

Reporting Bugs
##############

We use `GitHub issues`_ to track bugs, feature requests, and discussion threads.

If you'd like to report a bug or propose an enhancement, please:

1. Check if a similar issue or pull request already exists.
2. Include clear steps to reproduce the issue (if applicable).
3. Format code snippets using double colons (``::``) and indentation.
4. Share your environment details (e.g., OS, Python version, MissMecha version, numpy, scikit-learn).

.. _GitHub issues: https://github.com/echoid/MissMecha/issues


Improving Code or Documentation
###############################

All contributions to MissMecha should be linked to a `GitHub issue`_.  
If you notice something you'd like to improve but no issue exists yet, feel free to open one. That gives others the opportunity to discuss, offer context, or join forces.

Once your idea is discussed, you can:

- Leave a comment to "claim" the issue if you’re working on it.
- Create a pull request (PR) when your change is ready.

Pull Requests
#############

All contributions should be submitted as pull requests (PRs). The process is as follows:

1. **Fork** the MissMecha repository.
2. **Create a new branch** for your changes.
3. Implement your changes and **commit regularly**.
4. Push to your fork and **open a pull request** to the main repo.

See GitHub’s documentation on `creating a pull request from a fork`_ if needed.

.. _creating a pull request from a fork: https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/creating-a-pull-request-from-a-fork

Pull Request Checklist
######################

We recommend the following best practices for submitting PRs:

- Follow the `PEP8`_ style guide (max line length can be 100 characters).
- Include the issue number in your PR description to link discussions.
- Mark your PR as **"Draft"** if it’s still work-in-progress.
- Add tests for new functionality or bug fixes.
- Update or add documentation when appropriate.
- Include a short example if your feature affects user-facing behavior.

High-coverage tests and clear docs are essential for new features to be accepted.

.. _PEP8: https://www.python.org/dev/peps/pep-0008/
