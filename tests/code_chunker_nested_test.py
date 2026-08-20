import ast
import textwrap

from langchain_core.documents import Document


def _get_direct_nested_functions(
    node: ast.FunctionDef | ast.AsyncFunctionDef,
) -> list[ast.FunctionDef | ast.AsyncFunctionDef]:
    """
    Return functions directly nested inside the given function/method.
    """
    nested_functions = []

    for child in node.body:
        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
            nested_functions.append(child)

    return nested_functions


def _get_source_without_nested_functions(
    source_code: str,
    node: ast.FunctionDef | ast.AsyncFunctionDef,
) -> str:
    """
    Extract a function/method while replacing direct nested function
    implementations with structural comments.
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
            f"Could not extract source for node at line {node.lineno}"
        )

    # Trim the final line to the AST node's end column.
    selected_lines[-1] = selected_lines[-1][:end_col]

    nested_functions = _get_direct_nested_functions(node)

    # Replace nested function implementations from bottom to top.
    for nested_node in reversed(nested_functions):
        relative_start = nested_node.lineno - start_line
        relative_end = nested_node.end_lineno - start_line + 1

        if relative_start < 0:
            continue

        if relative_end > len(selected_lines):
            continue

        original_line = selected_lines[relative_start]

        indentation = len(original_line) - len(
            original_line.lstrip()
        )

        replacement = (
            " " * indentation
            + f"# nested function: {nested_node.name}\n"
        )

        selected_lines[relative_start:relative_end] = [
            replacement
        ]

    return textwrap.dedent("".join(selected_lines))


class ExperimentalCodeChunkVisitor(ast.NodeVisitor):
    """
    Experimental AST chunker.

    Each function/method remains its own chunk.

    For a parent function/method containing directly nested functions,
    the nested function implementations are replaced by a structural
    comment in the parent chunk.

    The nested functions themselves are still emitted as complete chunks.
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

    def visit_ClassDef(
        self,
        node: ast.ClassDef,
    ) -> None:
        self.class_stack.append(node.name)

        self.generic_visit(node)

        self.class_stack.pop()

    def visit_FunctionDef(
        self,
        node: ast.FunctionDef,
    ) -> None:
        self._add_function_chunk(node)

        self.function_stack.append(node.name)

        self.generic_visit(node)

        self.function_stack.pop()

    def visit_AsyncFunctionDef(
        self,
        node: ast.AsyncFunctionDef,
    ) -> None:
        self._add_function_chunk(node)

        self.function_stack.append(node.name)

        self.generic_visit(node)

        self.function_stack.pop()

    def _add_function_chunk(
        self,
        node: ast.FunctionDef | ast.AsyncFunctionDef,
    ) -> None:
        if self.class_stack:
            chunk_type = "method"
            class_name = ".".join(self.class_stack)
        else:
            chunk_type = "function"
            class_name = None

        qualified_parts = [
            *self.class_stack,
            *self.function_stack,
            node.name,
        ]

        qualified_name = ".".join(qualified_parts)

        module_name = (
            self.base_metadata["source"]
            .replace("/", ".")
            .removesuffix(".py")
        )

        page_content = _get_source_without_nested_functions(
            source_code=self.source_code,
            node=node,
        )

        metadata = self.base_metadata.copy()

        metadata.update(
            {
                "chunk_type": chunk_type,
                "module_name": module_name,
                "class_name": class_name,
                "function_name": node.name,
                "qualified_name": qualified_name,
            }
        )

        if class_name:
            header = (
                f"Module: {module_name}\n"
                f"Class: {class_name}\n"
                f"Method: {node.name}\n\n"
            )
        else:
            header = (
                f"Module: {module_name}\n"
                f"Function: {node.name}\n\n"
            )

        self.chunks.append(
            Document(
                page_content=header + page_content,
                metadata=metadata,
            )
        )


def experimental_chunk_python_documents(
    documents: list[Document],
) -> list[Document]:
    """
    Chunk Python documents using AST structure while removing
    direct nested-function implementations from parent chunks.
    """
    chunks = []

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

        visitor = ExperimentalCodeChunkVisitor(
            source_code=source_code,
            base_metadata=document.metadata,
        )

        visitor.visit(tree)

        chunks.extend(visitor.chunks)

    return chunks