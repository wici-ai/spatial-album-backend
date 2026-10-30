"""Optional host scheduler integration; standalone deployments need no adapter."""
import importlib
import os


def controlled_http_task(service, paths, *, finish):
    """Delegate admission and GPU completion to an explicitly selected host module.

    Missing configuration means standalone execution. An explicitly configured
    adapter must load successfully: a Box must never silently bypass admission.
    """
    module_name = os.environ.get("WICI_REQUEST_CONTROL_MODULE", "").strip()
    if not module_name:
        return lambda handler: handler
    adapter = getattr(importlib.import_module(module_name), "controlled_http_task")
    if not callable(adapter):
        raise TypeError(f"{module_name}.controlled_http_task must be callable")
    return adapter(service, paths, finish=finish)
