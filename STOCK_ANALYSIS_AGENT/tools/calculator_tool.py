# Advanced Calculator Tool for LangGraph
# By: [Your Name]

from langgraph.graph import StateGraph
from typing import Dict, List, Union
import ast
import operator
import re
import math

class AdvancedCalculator:
    """Handles complex math operations with safety checks"""
    
    def _init_(self):
        # Allowed operations and functions
        self.allowed_ops = {
            # Basic math
            ast.Add: operator.add,
            ast.Sub: operator.sub,
            ast.Mult: operator.mul,
            ast.Div: operator.truediv,
            ast.Pow: operator.pow,
            ast.Mod: operator.mod,
            ast.USub: operator.neg,
            ast.UAdd: operator.pos,
            
            # Comparisons
            ast.Eq: operator.eq,
            ast.NotEq: operator.ne,
            ast.Lt: operator.lt,
            ast.LtE: operator.le,
            ast.Gt: operator.gt,
            ast.GtE: operator.ge,
        }
        
        self.allowed_funcs = {
            # Math functions
            'sqrt': math.sqrt,
            'log': math.log10,
            'ln': math.log,
            'exp': math.exp,
            'sin': math.sin,
            'cos': math.cos,
            'tan': math.tan,
            'abs': abs,
            'round': round,
            
            # Stats functions
            'avg': lambda *args: sum(args)/len(args),
            'min': min,
            'max': max,
            'sum': sum,
        }
        
        # Constants
        self.constants = {
            'pi': math.pi,
            'e': math.e
        }
    
    def validate_input(self, expr: str) -> bool:
        """Check for allowed characters and patterns"""
        pattern = r'^[a-zA-Z0-9+\-*/().% ,_!<>=&|]+$'
        return bool(re.match(pattern, expr))
    
    def evaluate(self, node) -> Union[float, bool]:
        """Recursively evaluate AST nodes"""
        if isinstance(node, ast.Constant):
            return node.value
        elif isinstance(node, ast.Num):
            return node.n
        elif isinstance(node, ast.Name):
            # Handle constants
            if node.id in self.constants:
                return self.constants[node.id]
            raise ValueError(f"Unknown constant: {node.id}")
        elif isinstance(node, ast.BinOp):
            left = self.evaluate(node.left)
            right = self.evaluate(node.right)
            op = self.allowed_ops.get(type(node.op))
            if op is None:
                raise ValueError(f"Disallowed operation: {type(node.op)._name_}")
            return op(left, right)
        elif isinstance(node, ast.UnaryOp):
            operand = self.evaluate(node.operand)
            op = self.allowed_ops.get(type(node.op))
            if op is None:
                raise ValueError(f"Disallowed operation: {type(node.op)._name_}")
            return op(operand)
        elif isinstance(node, ast.Compare):
            left = self.evaluate(node.left)
            results = []
            for op, comp in zip(node.ops, node.comparators):
                right = self.evaluate(comp)
                op_func = self.allowed_ops.get(type(op))
                if op_func is None:
                    raise ValueError(f"Disallowed comparison: {type(op)._name_}")
                results.append(op_func(left, right))
                left = right
            return all(results)
        elif isinstance(node, ast.Call):
            # Handle function calls
            if not isinstance(node.func, ast.Name):
                raise ValueError("Function calls must use named functions")
            
            func_name = node.func.id
            if func_name not in self.allowed_funcs:
                raise ValueError(f"Disallowed function: {func_name}")
            
            args = [self.evaluate(arg) for arg in node.args]
            return self.allowed_funcs[func_name](*args)
        else:
            raise ValueError(f"Unsupported node type: {type(node)._name_}")

    def calculate(self, expr: str) -> Union[float, bool]:
        """Main calculation method"""
        if not self.validate_input(expr):
            raise ValueError("Invalid characters in expression")
        
        tree = ast.parse(expr, mode='eval')
        return self.evaluate(tree.body)

# LangGraph node that uses the advanced calculator
def advanced_math_node(state: Dict) -> Dict:
    """
    Solves complex math problems. Supports:
    - Basic arithmetic: 2*(3+4)
    - Scientific functions: sqrt(16) + sin(30)
    - Comparisons: 5 > 3 and 2 < 4
    - Statistics: avg(1,2,3,4)
    - Constants: pi * 2
    """
    problem = state.get('problem')
    if not problem:
        return {'result': None, 'error': 'No problem provided'}
    
    try:
        calc = AdvancedCalculator()
        result = calc.calculate(problem)
        return {'result': result, 'error': None}
    except Exception as e:
        return {'result': None, 'error': str(e)}

# Build the workflow
workflow = StateGraph(state_schema=Dict)
workflow.add_node("math_processor", advanced_math_node)
workflow.set_entry_point("math_processor")
workflow.set_finish_point("math_processor")
math_app = workflow.compile()

# Test examples
if __name__== "__main__":
    tests = [
        "2 * (3 + 4)",
        "sqrt(16) + sin(radians(30))",
        "pi * 5^2",
        "avg(10, 20, 30, 40)",
        "5 > 3 and 2 < 4",
        "log(100) + ln(e^3)",
        "round(pi, 3)"
    ]
    
    for test in tests:
        print(f"\nProblem: {test}")
        result = math_app.invoke({'problem': test})
        if result['error']:
            print(f"Error: {result['error']}")
        else:
            print(f"Result: {result['result']}")