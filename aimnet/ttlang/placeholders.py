from aimnet.ttlang.errors import DerivativeNotImplementedError


def forces_not_implemented(*_args: object, **_kwargs: object) -> None:
    raise DerivativeNotImplementedError(
        "Forces are not implemented for the ttlang backend. Use reference mode or PyTorch AIMNet2."
    )


def hessian_not_implemented(*_args: object, **_kwargs: object) -> None:
    raise DerivativeNotImplementedError(
        "Hessian is not implemented for the ttlang backend. Use reference mode or PyTorch AIMNet2."
    )
