def test_binding_importable_unconditionally():
    """The module should at least import without a network file; instantiating
    a worker is what requires a GPU and a .cnet network."""
    import _inference_cpp  # noqa: F401
