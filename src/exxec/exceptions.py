class NotInitializedError(RuntimeError):
    """Raised when the executor is not initialized."""

    def __init__(self, name: str):
        super().__init__(f"Environmment is not initialized: {name}.")
