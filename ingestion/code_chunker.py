import ast
import textwrap
from pathlib import Path

from langchain_core.documents import Document


def _get_module_name(source: str) -> str:
    """
    Convert a relative Python source path into a dotted module name.

    Example:
        checkpoint/serde/jsonplus.py
        -> checkpoint.serde.jsonplus
    """
    return Path(source).with_suffix("").as_posix().replace("/", ".")


def _get_source_segment(
    source_code: str,
    node: ast.FunctionDef | ast.AsyncFunctionDef,
) -> str:
    """
    Extract and normalize the original source code for a function
    or method, including decorators when present.
    """
    lines = source_code.splitlines(keepends=True)

    if node.decorator_list:
        start_line = min(
            decorator.lineno
            for decorator in node.decorator_list
        )
    else:
        start_line = node.lineno

    end_line = node.end_lineno
    end_col = node.end_col_offset

    selected_lines = lines[start_line - 1:end_line]

    if not selected_lines:
        raise ValueError(
            f"Could not extract source for AST node at line {node.lineno}"
        )

    selected_lines[-1] = selected_lines[-1][:end_col]

    source_segment = "".join(selected_lines)

    return textwrap.dedent(source_segment)


def _build_code_content(
    source_code: str,
    node: ast.FunctionDef | ast.AsyncFunctionDef,
    module_name: str,
    class_name: str | None,
    function_name: str,
) -> str:
    """
    Build the retrieval content for a function or method.

    A small structural header is prepended to the original source
    so lexical retrieval can benefit from module, class, and function
    names.
    """
    if class_name:
        structural_context = (
            f"Module: {module_name}\n"
            f"Class: {class_name}\n"
            f"Method: {function_name}\n"
        )
    else:
        structural_context = (
            f"Module: {module_name}\n"
            f"Function: {function_name}\n"
        )

    source_segment = _get_source_segment(
        source_code=source_code,
        node=node,
    )

    return f"{structural_context}\n{source_segment}"


class _CodeChunkVisitor(ast.NodeVisitor):
    """
    Traverse a Python AST and create one chunk per function/method.

    Classes themselves are not emitted as chunks. Their methods are.

    Nested functions and nested classes are still traversed so that
    their functions and methods are discovered.
    """

    def __init__(
        self,
        source_code: str,
        base_metadata: dict,
    ) -> None:
        self.source_code = source_code
        self.base_metadata = base_metadata
        self.chunks: list[Document] = []

        self.class_stack: list[str] = []
        self.function_stack: list[str] = []

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        """
        Track class context and continue traversing its contents.
        """
        self.class_stack.append(node.name)

        self.generic_visit(node)

        self.class_stack.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        """
        Create a chunk for a normal function or method, then continue
        traversal so nested functions and classes are discovered.
        """
        self._add_function_chunk(node)

        self.function_stack.append(node.name)

        self.generic_visit(node)

        self.function_stack.pop()

    def visit_AsyncFunctionDef(
        self,
        node: ast.AsyncFunctionDef,
    ) -> None:
        """
        Create a chunk for an async function or method, then continue
        traversal so nested functions and classes are discovered.
        """
        self._add_function_chunk(node)

        self.function_stack.append(node.name)

        self.generic_visit(node)

        self.function_stack.pop()

    def _add_function_chunk(
        self,
        node: ast.FunctionDef | ast.AsyncFunctionDef,
    ) -> None:
        """
        Convert a function/method AST node into a LangChain Document.
        """
        function_name = node.name

        if self.class_stack:
            chunk_type = "method"
            class_name = ".".join(self.class_stack)
        else:
            chunk_type = "function"
            class_name = None

        qualified_parts = [
            *self.class_stack,
            *self.function_stack,
            function_name,
        ]

        qualified_name = ".".join(qualified_parts)

        module_name = _get_module_name(
            self.base_metadata["source"]
        )

        page_content = _build_code_content(
            source_code=self.source_code,
            node=node,
            module_name=module_name,
            class_name=class_name,
            function_name=function_name,
        )

        metadata = self.base_metadata.copy()

        metadata.update(
            {
                "chunk_type": chunk_type,
                "module_name": module_name,
                "class_name": class_name,
                "function_name": function_name,
                "qualified_name": qualified_name,
            }
        )

        self.chunks.append(
            Document(
                page_content=page_content,
                metadata=metadata,
            )
        )


def chunk_python_documents(
    documents: list[Document],
) -> list[Document]:
    """
    Chunk Python implementation documents into AST-level semantic units.

    Each function or method becomes one Document.

    Classes are not emitted as standalone chunks.

    Oversized functions/methods are intentionally kept as a single
    chunk for now. Their size distribution will be measured after
    chunking and used to determine whether further splitting is needed.

    Args:
        documents: Full Python-file Documents.

    Returns:
        A list of semantic code Documents.
    """
    chunks: list[Document] = []

    for document in documents:
        source_code = document.page_content

        if not source_code.strip():
            continue

        try:
            tree = ast.parse(source_code)
        except SyntaxError as exc:
            raise ValueError(
                f"Failed to parse Python source: "
                f"{document.metadata.get('source')}"
            ) from exc

        visitor = _CodeChunkVisitor(
            source_code=source_code,
            base_metadata=document.metadata,
        )

        visitor.visit(tree)

        chunks.extend(visitor.chunks)

    return chunks