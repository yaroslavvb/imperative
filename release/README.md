Tools for releasing/packaging imperative mode.

To test the release:

    ./release/make_package.sh
    tests/run_all_tests.sh

    # to test single file release
    cd release
    python
    import imperative
    imperative.self_test()
