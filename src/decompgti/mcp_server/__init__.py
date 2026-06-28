"""MCP server package for graph tools used by DecompGTI."""


def build_server():
	from .server import build_server as _build_server

	return _build_server()


__all__ = ["build_server"]
