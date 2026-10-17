"""Tiny helpers to build notebook cells."""
import nbformat


def md(text: str):
    return nbformat.v4.new_markdown_cell(text.strip("\n"))


def code(text: str):
    return nbformat.v4.new_code_cell(text.strip("\n"))
