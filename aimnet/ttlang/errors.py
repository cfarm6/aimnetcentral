"""Exceptions for the TT-Lang AIMNet2 backend."""


class TTBackendUnavailableError(RuntimeError):
    """Raised when a Tenstorrent backend is requested but cannot be initialized."""


class DerivativeNotImplementedError(NotImplementedError):
    """Raised when forces or Hessian are requested on the ttlang backend."""
