class ModelNotLoadedError(RuntimeError):
    """Raised when prediction is attempted before the model is loaded."""