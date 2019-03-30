class EmptyTensorException(Exception):
    """
    Prevent possible error when dealing with empty tensor.
    """
    pass


class MultiFaceException(Exception):
    """
    Detect multiple face in a single image.
    """
    pass


class NoSuchNameException(Exception):
    """
    Raised when specify a non-existing attribute by string
    """
    pass

