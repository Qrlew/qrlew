# Testing Qrlew

## Launch tests

Simply launch:
`cargo test`

## Testing on many linuses and architectures

Qrlew will usually be used with its python wrapper that is built against the architectures in: https://github.com/PyO3/maturin-action
Run the tests in one of the architectures if needed:
`docker run -it --rm quay.io/pypa/manylinux_2_28_x86_64:latest`

