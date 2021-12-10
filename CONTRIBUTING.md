# Pull Requests

Intel HE-FPGA welcomes pull requests from external contributors to the `main` branch.

Before contributing, please make sure all tests pass, and the pre-commit formatting and linter checks are clean.

Please sign your commits before making a pull request. See instructions [here](https://docs.github.com/en/github/authenticating-to-github/managing-commit-signature-verification/signing-commits) for how to sign commits.

### Known Issues ###

* ```Executable `cpplint` not found```

  Make sure you install cpplint: ```pip install cpplint```.
  If you install `cpplint` locally, make sure to add it to your `PATH`.

* ```/bin/sh: 1: pre-commit: not found```

  Install `pre-commit`. More info at https://pre-commit.com/.
