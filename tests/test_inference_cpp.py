def test_binding_importable_unconditionally():
    """The module should at least import even without a TRT engine; instantiating
    a worker is what requires GPU/TRT."""
    import _inference_cpp  # noqa: F401
