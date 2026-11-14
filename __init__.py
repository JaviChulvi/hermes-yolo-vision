"""Hermes directory-plugin entry point; registration never loads model weights."""


def register(ctx):
    from .hermes_yolo_vision.plugin import register as register_plugin

    register_plugin(ctx)

__all__ = ["register"]
