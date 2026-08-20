import ast
import textwrap

from langchain_core.documents import Document


def _get_nested_functions(
    node: ast.FunctionDef | ast.AsyncFunctionDef,
) -> list[ast.FunctionDef | ast.AsyncFunctionDef]:
    """
    Return the outermost nested functions contained inside the given
    function/method.

    If a nested function contains another nested function, only the
    outer nested function is returned. Removing its source will
    automatically remove its descendants as well.
    """
    nested_functions = []

    class NestedFunctionVisitor(ast.NodeVisitor):
        def __init__(self) -> None:
            self.found_nested_function = False

        def visit_FunctionDef(
            self,
            child: ast.FunctionDef,
        ) -> None:
            if not self.found_nested_function:
                nested_functions.append(child)

            # Do not visit descendants once this function has been
            # identified. Its complete source range will be removed.
            self.found_nested_function = True

        def visit_AsyncFunctionDef(
            self,
            child: ast.AsyncFunctionDef,
        ) -> None:
            if not self.found_nested_function:
                nested_functions.append(child)

            # Do not visit descendants once this function has been
            # identified.
            self.found_nested_function = True

    # We need to inspect every part of the function body because
    # nested functions can be inside if/try/with/for blocks.
    class RecursiveNestedFunctionVisitor(ast.NodeVisitor):
        def __init__(self) -> None:
            self.inside_nested_function = False

        def visit_FunctionDef(
            self,
            child: ast.FunctionDef,
        ) -> None:
            if self.inside_nested_function:
                return

            nested_functions.append(child)

        def visit_AsyncFunctionDef(
            self,
            child: ast.AsyncFunctionDef,
        ) -> None:
            if self.inside_nested_function:
                return

            nested_functions.append(child)

        def generic_visit(self, child: ast.AST) -> None:
            if isinstance(
                child,
                (ast.FunctionDef, ast.AsyncFunctionDef),
            ):
                # Function definitions are handled by the methods above.
                return

            super().generic_visit(child)

    # The visitor above needs to distinguish the root node from
    # nested functions, so walk the root's body manually.
    class DescendantVisitor(ast.NodeVisitor):
        def __init__(self) -> None:
            self.results: list[
                ast.FunctionDef | ast.AsyncFunctionDef
            ] = []

        def visit_FunctionDef(
            self,
            child: ast.FunctionDef,
        ) -> None:
            self.results.append(child)

        def visit_AsyncFunctionDef(
            self,
            child: ast.AsyncFunctionDef,
        ) -> None:
            self.results.append(child)

    # Find all descendants.
    visitor = DescendantVisitor()

    for child in node.body:
        visitor.visit(child)

    # We only want outermost nested functions.
    #
    # If function A contains function B, A's source range contains B's
    # source range. Therefore B does not need a separate replacement.
    for candidate in visitor.results:
        is_nested_inside_another = False

        for other in visitor.results:
            if candidate is other:
                continue

            if (
                other.lineno <= candidate.lineno
                and other.end_lineno >= candidate.end_lineno
            ):
                is_nested_inside_another = True
                break

        if not is_nested_inside_another:
            nested_functions.append(candidate)

    # Remove duplicates caused by the different visitor paths.
    unique_functions = {
        id(function): function
        for function in nested_functions
    }

    return list(unique_functions.values())


def _get_source_without_nested_functions(
    source_code: str,
    node: ast.FunctionDef | ast.AsyncFunctionDef,
) -> str:
    """
    Extract a function/method while replacing all nested function
    implementations with structural comments.

    Nested functions themselves remain independently indexed.
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

    selected_lines[-1] = selected_lines[-1][:end_col]

    nested_functions = _get_nested_functions(node)

    # Replace from bottom to top so source positions remain valid.
    nested_functions.sort(
        key=lambda function: function.lineno,
        reverse=True,
    )

    for nested_node in nested_functions:
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
    Experimental AST code chunker.

    Every function/method remains an independent chunk.

    All nested function implementations are removed from the parent
    chunk and replaced with structural comments.

    Nested functions themselves remain complete independent chunks.
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
    nested-function implementations from parent chunks.
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