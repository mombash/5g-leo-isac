"""Adapt four-attempt buffers and controllers to one-second recordings."""
import ast

class ShortData(ast.NodeTransformer):

    def visit_Constant(self, node):
        if type(node.value) in (int, float) and node.value in (4, 4.01, 200):
            return ast.copy_location(ast.Constant({4: 1, 4.01: 1.01, 200: 50}[node.value]), node)
        return node

class ShortScheduler(ast.NodeTransformer):

    def visit_Constant(self, node):
        if type(node.value) is int and node.value in (200, 800):
            return ast.copy_location(ast.Constant({200: 50, 800: 200}[node.value]), node)
        return node

    def visit_Compare(self, node):
        node = self.generic_visit(node)
        if isinstance(node.left, ast.Call) and isinstance(node.left.func, ast.Name) and (node.left.func.id == 'len') and (ast.unparse(node.left.args[0]) == 'self.done'):
            for n in node.comparators:
                if isinstance(n, ast.Constant) and n.value == 4:
                    n.value = 1
        return node

    def visit_Call(self, node):
        node = self.generic_visit(node)
        if isinstance(node.func, ast.Name) and node.func.id == 'range':
            if len(node.args) == 1 and isinstance(node.args[0], ast.Constant) and (node.args[0].value == 4):
                node.args[0].value = 1
        for kw in node.keywords:
            if kw.arg == 'max_workers' and isinstance(kw.value, ast.Constant) and (kw.value.value == 4):
                kw.value.value = 1
        return node
