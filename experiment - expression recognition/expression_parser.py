"""
Turns a sequence of classified symbols into an expression string
(including parentheses) and evaluates it safely - no raw eval(), which
would execute arbitrary code if anything unexpected ended up in the string.

Example: symbols ['(','2','3','4','7','x','6','5','8','1',')','-',
                   '(','3','4','2','x','1','8',')']
      -> "(2347*6581)-(342*18)"
      -> evaluated respecting normal order of operations AND parentheses
         (Python's own parser handles the grouping, so this needs no
         special-case code for '(' / ')')
"""

import ast
import operator

SYMBOL_TO_CHAR = {
    "0": "0", "1": "1", "2": "2", "3": "3", "4": "4",
    "5": "5", "6": "6", "7": "7", "8": "8", "9": "9",
    "+": "+", "-": "-", "x": "*", "/": "/", "(": "(", ")": ")",
}

_ALLOWED_BINOPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
}
_ALLOWED_UNARYOPS = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}


def build_expression_string(symbols: list) -> str:
    return "".join(SYMBOL_TO_CHAR[s] for s in symbols)


class ExpressionError(ValueError):
    pass


def safe_eval(expr: str) -> float:
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError as e:
        raise ExpressionError(f"Could not parse recognized expression '{expr}': {e}")

    def _eval(node):
        if isinstance(node, ast.Expression):
            return _eval(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_BINOPS:
            left, right = _eval(node.left), _eval(node.right)
            if isinstance(node.op, ast.Div) and right == 0:
                raise ExpressionError("Division by zero in recognized expression")
            return _ALLOWED_BINOPS[type(node.op)](left, right)
        if isinstance(node, ast.UnaryOp) and type(node.op) in _ALLOWED_UNARYOPS:
            return _ALLOWED_UNARYOPS[type(node.op)](_eval(node.operand))
        raise ExpressionError(f"Disallowed expression element: {ast.dump(node)}")

    return _eval(tree.body) if isinstance(tree, ast.Expression) else _eval(tree)


def recognize_and_evaluate(symbols: list):
    expr = build_expression_string(symbols)
    result = safe_eval(expr)
    return expr, result
