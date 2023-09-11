# Contribution Guide

Welcome to the Fiddler Auditor contribution guide. Firstly, we'd like to
thank you for considering to contribute to the Auditor! 

This guide captures the following

- Ways to Contribute
- Contributing Code
- Setting up DEV Environment
- Running tests locally
- Pull Request (PR) approval process

## Ways to Contribute

You can contribute in the following ways

- 🚩 File Bug reports
- 🤔 Raise Issues for feature requests and improvements
- 👩‍💻 Contribute code via PRs. Some of the ways in which you can contribute
    - 📝 Add an example Notebook
    - 📚 Improve documentation including this guide!
    - 🐞 Work on a feature/bug
    - 🟩 Improve unit-tests

## Contributing Code

- We follow the ["fork and pull request workflow"](https://docs.github.com/en/get-started/quickstart/contributing-to-projects) for code contribution.
- Once you make your changes and submit a pull-request (PR). We automatically run a 
series of checks and require one of the Maintainers to approve the request for the 
changes to land.

## Setting-up DEV environment

We recommend setting up a separate python/conda virtual environment and then 
run the following command

```bash
pip install .[test]
```

## Running tests locally

To run the unit-tests use the following command

```bash
make test
```

Note that some of the tests rely on access to OpenAI APIs. This reliance will 
be removed but for now you'll need to set API key as follows

```bash
export OPENAI_API_KEY="..."
```

To run the lint check use the command

```bash
make lint
```

## Pull Request (PR) approval process

Once you are done with your code-changes on your branch open a pull-request 
for the maintainers to review and comment. For a PR to be approved following 
conditions need to be satifised

- Lint Check
- Unit-tests across different versions of Python
- Approval from atleast one maintainer