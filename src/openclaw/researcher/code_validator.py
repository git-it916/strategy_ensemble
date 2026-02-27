"""
Code Validator

AST-based safety validation for generated alpha code.
Checks imports, dangerous calls, class structure, and interface compliance.
"""

from __future__ import annotations

import ast
import logging
from dataclasses import dataclass, field

from src.openclaw.config import DANGEROUS_BUILTINS, DANGEROUS_MODULES, IMPORT_WHITELIST

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Result of code validation."""

    is_valid: bool = True
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


class CodeValidator:
    """
    AST-based safety validation for generated alpha code.

    Checks:
    1. Code parses as valid Python
    2. Only whitelisted imports
    3. No dangerous function calls (eval, exec, open, subprocess, etc.)
    4. Proper class structure (subclasses BaseAlpha)
    5. Required methods implemented (fit, generate_signals)
    6. No global state mutation
    """

    def __init__(
        self,
        import_whitelist: frozenset[str] | None = None,
        dangerous_modules: frozenset[str] | None = None,
        dangerous_builtins: frozenset[str] | None = None,
    ):
        self.import_whitelist = import_whitelist or IMPORT_WHITELIST
        self.dangerous_modules = dangerous_modules or DANGEROUS_MODULES
        self.dangerous_builtins = dangerous_builtins or DANGEROUS_BUILTINS

    def validate(self, code: str) -> ValidationResult:
        """
        Run all validation checks on generated code.

        Args:
            code: Python source code string

        Returns:
            ValidationResult with is_valid, errors, warnings
        """
        result = ValidationResult()

        # 1. Parse AST
        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            result.is_valid = False
            result.errors.append(f"SyntaxError: {e}")
            return result

        # 2. Check imports
        import_errors = self._check_imports(tree)
        result.errors.extend(import_errors)

        # 3. Check dangerous calls
        call_errors = self._check_dangerous_calls(tree)
        result.errors.extend(call_errors)

        # 4. Check class structure
        struct_errors, struct_warnings = self._check_class_structure(tree)
        result.errors.extend(struct_errors)
        result.warnings.extend(struct_warnings)

        # 5. Check for global state mutation
        global_errors = self._check_global_state(tree)
        result.warnings.extend(global_errors)

        result.is_valid = len(result.errors) == 0

        if result.is_valid:
            logger.info("Code validation passed")
        else:
            logger.warning(f"Code validation failed: {result.errors}")

        return result

    def _check_imports(self, tree: ast.Module) -> list[str]:
        """Check that all imports are from the whitelist."""
        errors = []

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    module = alias.name.split(".")[0]
                    if module in self.dangerous_modules:
                        errors.append(
                            f"Forbidden import: '{alias.name}' "
                            f"(module '{module}' is dangerous)"
                        )
                    elif module not in self.import_whitelist:
                        errors.append(
                            f"Non-whitelisted import: '{alias.name}' "
                            f"(add '{module}' to whitelist if safe)"
                        )

            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    module = node.module.split(".")[0]
                    if module in self.dangerous_modules:
                        errors.append(
                            f"Forbidden import: 'from {node.module}' "
                            f"(module '{module}' is dangerous)"
                        )
                    elif module not in self.import_whitelist:
                        errors.append(
                            f"Non-whitelisted import: 'from {node.module}' "
                            f"(add '{module}' to whitelist if safe)"
                        )

        return errors

    def _check_dangerous_calls(self, tree: ast.Module) -> list[str]:
        """Check for dangerous function calls."""
        errors = []

        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                func_name = self._get_call_name(node)
                if func_name in self.dangerous_builtins:
                    errors.append(
                        f"Forbidden call: '{func_name}()' "
                        f"at line {node.lineno}"
                    )

                # Check for os.system, subprocess.run, etc.
                if isinstance(node.func, ast.Attribute):
                    if isinstance(node.func.value, ast.Name):
                        module_name = node.func.value.id
                        if module_name in self.dangerous_modules:
                            errors.append(
                                f"Forbidden call: '{module_name}.{node.func.attr}()' "
                                f"at line {node.lineno}"
                            )

        return errors

    def _check_class_structure(
        self, tree: ast.Module
    ) -> tuple[list[str], list[str]]:
        """Check that code defines a proper BaseAlpha subclass."""
        errors = []
        warnings = []

        # Find class definitions
        classes = [
            node for node in ast.walk(tree) if isinstance(node, ast.ClassDef)
        ]

        if not classes:
            errors.append("No class definition found")
            return errors, warnings

        # Check for BaseAlpha inheritance
        has_base_alpha = False
        for cls in classes:
            for base in cls.bases:
                base_name = self._get_node_name(base)
                if base_name in ("BaseAlpha", "BaseMLAlpha"):
                    has_base_alpha = True

                    # Check required methods
                    method_names = {
                        node.name
                        for node in ast.walk(cls)
                        if isinstance(node, ast.FunctionDef)
                    }

                    if "fit" not in method_names:
                        errors.append(
                            f"Class '{cls.name}' missing required method 'fit()'"
                        )
                    if "generate_signals" not in method_names:
                        errors.append(
                            f"Class '{cls.name}' missing required method "
                            f"'generate_signals()'"
                        )

                    # Check __init__ calls super().__init__
                    if "__init__" in method_names:
                        init_node = next(
                            n for n in ast.walk(cls)
                            if isinstance(n, ast.FunctionDef) and n.name == "__init__"
                        )
                        has_super = any(
                            isinstance(n, ast.Call)
                            and self._is_super_init(n)
                            for n in ast.walk(init_node)
                        )
                        if not has_super:
                            warnings.append(
                                f"Class '{cls.name}.__init__' may not call "
                                f"super().__init__()"
                            )

        if not has_base_alpha:
            errors.append(
                "No class inheriting from BaseAlpha or BaseMLAlpha found"
            )

        return errors, warnings

    def _check_global_state(self, tree: ast.Module) -> list[str]:
        """Check for global state mutation (assignments outside classes/functions)."""
        warnings = []

        for node in tree.body:
            # Allow imports, class defs, function defs, and simple assignments
            if isinstance(node, (ast.Import, ast.ImportFrom, ast.ClassDef,
                                 ast.FunctionDef, ast.AsyncFunctionDef,
                                 ast.Expr)):
                continue

            if isinstance(node, ast.Assign):
                # Allow module-level constants (ALL_CAPS)
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        if not target.id.isupper() and not target.id.startswith("_"):
                            warnings.append(
                                f"Global mutable assignment: '{target.id}' "
                                f"at line {node.lineno}"
                            )

        return warnings

    @staticmethod
    def _get_call_name(node: ast.Call) -> str:
        """Get the function name from a Call node."""
        if isinstance(node.func, ast.Name):
            return node.func.id
        if isinstance(node.func, ast.Attribute):
            return node.func.attr
        return ""

    @staticmethod
    def _get_node_name(node: ast.expr) -> str:
        """Get name from various AST node types."""
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Attribute):
            return node.attr
        return ""

    @staticmethod
    def _is_super_init(node: ast.Call) -> bool:
        """Check if a Call node is super().__init__()."""
        if isinstance(node.func, ast.Attribute):
            if node.func.attr == "__init__":
                if isinstance(node.func.value, ast.Call):
                    if isinstance(node.func.value.func, ast.Name):
                        return node.func.value.func.id == "super"
        return False
