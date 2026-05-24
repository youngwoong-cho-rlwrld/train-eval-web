"""Shared Kubernetes transport error classifiers."""

_KUBECTL_TRANSPORT_TOKENS = (
    "failed calling webhook",
    "failed to call webhook",
    "admission-webhook",
    "error sending request",
    "cannot assign requested address",
    "connection refused",
    "i/o timeout",
    "tls handshake timeout",
    "context deadline exceeded",
    "internal error occurred",
    "no such host",
)

_KUBECTL_COMPLETED_POD_EXEC_TOKENS = (
    "cannot exec into a container in a completed pod",
    "current phase is succeeded",
    "current phase is failed",
)

_KUBECTL_EXEC_STREAM_TOKENS = (
    "unexpected eof",
    "error reading from error stream",
    "copying stderr failed",
    "copying stdout failed",
    "waiting for server to close stdin failed",
    *_KUBECTL_COMPLETED_POD_EXEC_TOKENS,
)


def is_kubectl_transport_error(message: str) -> bool:
    lower = message.lower()
    return any(token in lower for token in _KUBECTL_TRANSPORT_TOKENS)


def is_kubectl_exec_transport_error(message: str) -> bool:
    lower = message.lower()
    return is_kubectl_transport_error(message) or any(
        token in lower for token in _KUBECTL_EXEC_STREAM_TOKENS
    )


def is_completed_pod_exec_error(message: str) -> bool:
    lower = message.lower()
    return any(token in lower for token in _KUBECTL_COMPLETED_POD_EXEC_TOKENS)
