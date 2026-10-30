"""Entry point: python module/main.py <socket path> (viam-server starts this)."""
import asyncio

from viam.module.module import Module

from follow_service import ObjectFollowService  # noqa: F401  (importing registers the model)

if __name__ == "__main__":
    asyncio.run(Module.run_from_registry())
