Contributing Guide
==================

Contributions are what make the open source community such an amazing place to learn, inspire, and create. Any contributions you make are welcome and **greatly appreciated**.
Feel free to contribute any kind of function or enhancement.

A bug-fix or enhancement is delivered by using a pull request. A good pull request
should cover one bug-fix or enhancement feature. This ensures the change set is
easier to review.

Local setup is required. Make sure you:

* Fork the `SiaPy` repository into your account.
* Obtain the source by cloning it onto your development machine.
* Have local version up and running by following the [setup guide](docs/setup_guide.md).

To make contributions:

* Create a branch for local development:

    ```sh
    $ git checkout -b name-of-your-bugfix-or-feature
    ```

* Develop a fix or enhancement:
  * E.g. modify a class, method, function, module, etc.
  * The docs should be updated for anything but trivial bug fixes.

* Commit and push changes to your fork:

    ```sh
    $ git add .
    $ git commit -m "A detailed description of the changes."
    $ git push origin name-of-your-bugfix-or-feature
    ```

* Submit a pull request through the service website (e.g. Github).

    A pull request should preferably only have one commit upon the current master HEAD, (via rebases and squash).
