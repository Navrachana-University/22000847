import re
import subprocess
import os
from collections import namedtuple

# Common Token definition
Token = namedtuple('Token', ['type', 'value', 'line', 'col'])

#Lexical Analyzer
class HindiLexer:               #Lexical Grammar 
    token_specs = [
        ('PRINT', r'प्रिंट'),
        ('IF', r'यदि'),
        ('ELSE', r'नहीं_तो'),
        ('ELIF', r'या_तो'),
        ('WHILE', r'जबतक'),
        ('FOR', r'लिए'),
        ('BREAK', r'तोड़'),
        ('CONTINUE', r'जारी_रखो'),
        ('RETURN', r'वापस_दो'),
        ('FUNCTION', r'फ़ंक्शन'),
        ('ASSIGN', r'='),
        ('LPAREN', r'\('),
        ('RPAREN', r'\)'),
        ('LBRACE', r'\{'),
        ('RBRACE', r'\}'),
        ('COMMA', r','),
        ('NUMBER', r'\d+(\.\d+)?'),
        ('EXPONENT', r'\^'),
        ('PLUS', r'\+'),
        ('MINUS', r'-'),
        ('MULTIPLY', r'\*'),
        ('DIVIDE', r'/'),
        ('MODULUS', r'%'),
        ('EQUAL', r'=='),
        ('NOT_EQUAL', r'!='),
        ('LESS', r'<'),
        ('LESS_EQUAL', r'<='),
        ('GREATER', r'>'),
        ('GREATER_EQUAL', r'>='),
        ('STRING', r'"[^"]*"'),
        ('ID', r'[a-zA-Z\u0900-\u097F_][a-zA-Z\u0900-\u097F_\d]*'),
        ('SEMI', r';'),
        ('SKIP', r'[ \t\n]+'),
        ('COMMENT', r'//.*'),
        ('NEWLINE', r'\n'),
        ('MISMATCH', r'.'),
    ]

    def __init__(self, source):
        self.source = source
        self.pos = 0
        self.line = 1
        self.col = 1
        self.tokens = []
        self._token_re = re.compile(
            '|'.join(f'(?P<{pair[0]}>{pair[1]})' for pair in self.token_specs),
            re.UNICODE
        )

    def tokenize(self):
        while self.pos < len(self.source):
            char = self.source[self.pos]
            mo = self._token_re.match(self.source, self.pos)
            
            if not mo:
                raise ValueError(f'Unexpected character "{char}" at line {self.line}, column {self.col}')
            
            typ = mo.lastgroup
            val = mo.group(typ)
            
            if typ == 'NEWLINE':
                self.line += 1
                self.col = 1
            else:
                self.col += len(val)
            
            if typ == 'SKIP':
                self.pos = mo.end()
                continue
            
            if typ == 'COMMENT':
                self.pos = mo.end()
                continue
            
            if typ == 'MISMATCH':
                raise ValueError(f'Unexpected character "{val}" at line {self.line}, column {self.col}')
            
            self.pos = mo.end()
            self.tokens.append(Token(typ, val, self.line, self.col))
        
        self.tokens.append(Token('EOF', '', self.line, self.col))
        return self.tokens

#Parsing
class HindiParser:
    def __init__(self, tokens):
        self.tokens = iter(tokens)
        self.current_token = next(self.tokens, Token('EOF', '', 0, 0))
    
    def parse(self):
        return self.parse_program()
    
    #Syntactic Grammar
    def parse_program(self):
        statements = []
        while self.current_token.type != 'EOF':
            if self.current_token.type == 'PRINT':
                statements.append(self.parse_print())
            elif self.current_token.type == 'IF':
                statements.append(self.parse_if())
            elif self.current_token.type == 'ELSE':
                statements.append(self.parse_else())
            elif self.current_token.type == 'ELIF':
                statements.append(self.parse_elif())
            elif self.current_token.type == 'WHILE':
                statements.append(self.parse_while())
            elif self.current_token.type == 'FOR':
                statements.append(self.parse_for())
            elif self.current_token.type == 'BREAK':
                statements.append(self.parse_break())
            elif self.current_token.type == 'CONTINUE':
                statements.append(self.parse_continue())
            elif self.current_token.type == 'RETURN':
                statements.append(self.parse_return())
            elif self.current_token.type == 'FUNCTION':
                statements.append(self.parse_function())
            elif self.current_token.type == 'ID':
                statements.append(self.parse_assignment())
            else:
                raise SyntaxError(f'Unexpected token {self.current_token.value} at line {self.current_token.line}, column {self.current_token.col}')
        return {'type': 'program', 'body': statements}
    
    def parse_print(self):
        self.eat('PRINT')
        self.eat('LPAREN')
        expr = self.parse_expr()
        self.eat('RPAREN')
        self.eat('SEMI')
        return {'type': 'print', 'expression': expr}
    
    def parse_assignment(self):
        var_name = self.current_token.value
        line = self.current_token.line
        col = self.current_token.col
        self.eat('ID')
        self.eat('ASSIGN')
        expr = self.parse_expr()
        self.eat('SEMI')
        return {'type': 'assignment', 'target': var_name, 'value': expr, 'line': line, 'col': col}
    
    def parse_if(self):
        self.eat('IF')
        self.eat('LPAREN')
        condition = self.parse_expr()
        self.eat('RPAREN')
        body = self.parse_block()
        return {'type': 'if_statement', 'condition': condition, 'body': body}
    
    def parse_else(self):
        self.eat('ELSE')
        body = self.parse_block()
        return {'type': 'else_statement', 'body': body}
    
    def parse_elif(self):
        self.eat('ELIF')
        self.eat('LPAREN')
        condition = self.parse_expr()
        self.eat('RPAREN')
        body = self.parse_block()
        return {'type': 'elif_statement', 'condition': condition, 'body': body}
    
    def parse_while(self):
        self.eat('WHILE')
        self.eat('LPAREN')
        condition = self.parse_expr()
        self.eat('RPAREN')
        body = self.parse_block()
        return {'type': 'while_statement', 'condition': condition, 'body': body}
    
    def parse_for(self):
        self.eat('FOR')
        self.eat('LPAREN')
        init = self.parse_expr()
        self.eat('SEMI')
        condition = self.parse_expr()
        self.eat('SEMI')
        update = self.parse_expr()
        self.eat('RPAREN')
        body = self.parse_block()
        return {'type': 'for_statement', 'init': init, 'condition': condition, 'update': update, 'body': body}
    
    def parse_break(self):
        self.eat('BREAK')
        self.eat('SEMI')
        return {'type': 'break_statement'}
    
    def parse_continue(self):
        self.eat('CONTINUE')
        self.eat('SEMI')
        return {'type': 'continue_statement'}
    
    def parse_return(self):
        self.eat('RETURN')
        expr = self.parse_expr()
        self.eat('SEMI')
        return {'type': 'return_statement', 'expression': expr}
    
    def parse_function(self):
        self.eat('FUNCTION')
        name = self.current_token.value
        self.eat('ID')
        self.eat('LPAREN')
        params = []
        if self.current_token.type != 'RPAREN':
            params.append(self.current_token.value)
            self.eat('ID')
            while self.current_token.type == 'COMMA':
                self.eat('COMMA')
                params.append(self.current_token.value)
                self.eat('ID')
        self.eat('RPAREN')
        body = self.parse_block()
        return {'type': 'function_declaration', 'name': name, 'parameters': params, 'body': body}
    
    def parse_block(self):
        self.eat('LBRACE')
        statements = []
        while self.current_token.type != 'RBRACE':
            if self.current_token.type == 'PRINT':
                statements.append(self.parse_print())
            elif self.current_token.type == 'IF':
                statements.append(self.parse_if())
            elif self.current_token.type == 'ELSE':
                statements.append(self.parse_else())
            elif self.current_token.type == 'ELIF':
                statements.append(self.parse_elif())
            elif self.current_token.type == 'WHILE':
                statements.append(self.parse_while())
            elif self.current_token.type == 'FOR':
                statements.append(self.parse_for())
            elif self.current_token.type == 'BREAK':
                statements.append(self.parse_break())
            elif self.current_token.type == 'CONTINUE':
                statements.append(self.parse_continue())
            elif self.current_token.type == 'RETURN':
                statements.append(self.parse_return())
            elif self.current_token.type == 'FUNCTION':
                statements.append(self.parse_function())
            elif self.current_token.type == 'ID':
                statements.append(self.parse_assignment())
            else:
                raise SyntaxError(f'Unexpected token {self.current_token.value} at line {self.current_token.line}, column {self.current_token.col}')
        self.eat('RBRACE')
        return {'type': 'block', 'body': statements}
    
    def parse_expr(self):
        return self.parse_equality()
    
    def parse_equality(self):
        expr = self.parse_comparison()
        while self.current_token.type in ('EQUAL', 'NOT_EQUAL'):
            op = self.current_token.type
            self.eat(op)
            expr = {
                'type': 'binary_expression',
                'operator': op,
                'left': expr,
                'right': self.parse_comparison()
            }
        return expr
    
    def parse_comparison(self):
        expr = self.parse_term()
        while self.current_token.type in ('LESS', 'LESS_EQUAL', 'GREATER', 'GREATER_EQUAL'):
            op = self.current_token.type
            self.eat(op)
            expr = {
                'type': 'binary_expression',
                'operator': op,
                'left': expr,
                'right': self.parse_term()
            }
        return expr
    
    def parse_term(self):
        expr = self.parse_exponent()
        while self.current_token.type in ('PLUS', 'MINUS'):
            op = self.current_token.type
            self.eat(op)
            expr = {
                'type': 'binary_expression',
                'operator': op,
                'left': expr,
                'right': self.parse_exponent()
            }
        return expr
    
    def parse_exponent(self):
        expr = self.parse_factor()
        while self.current_token.type == 'EXPONENT':
            self.eat('EXPONENT')
            expr = {
                'type': 'binary_expression',
                'operator': 'EXPONENT',
                'left': expr,
                'right': self.parse_factor()
            }
        return expr
    
    def parse_factor(self):
        expr = self.parse_primary()
        while self.current_token.type in ('MULTIPLY', 'DIVIDE', 'MODULUS'):
            op = self.current_token.type
            self.eat(op)
            expr = {
                'type': 'binary_expression',
                'operator': op,
                'left': expr,
                'right': self.parse_primary()
            }
        return expr
    
    def parse_primary(self):
        token = self.current_token
        if token.type == 'MINUS':
            self.eat('MINUS')
            return {
                'type': 'unary_expression',
                'operator': '-',
                'argument': self.parse_primary()
            }
        elif token.type == 'NUMBER':
            self.eat('NUMBER')
            return {'type': 'number', 'value': float(token.value)}
        elif token.type == 'STRING':
            self.eat('STRING')
            return {'type': 'string', 'value': token.value.strip('"')}
        elif token.type == 'ID':
            name = token.value
            self.eat('ID')
            return {'type': 'identifier', 'name': name}
        elif token.type == 'LPAREN':
            self.eat('LPAREN')
            expr = self.parse_expr()
            self.eat('RPAREN')
            return expr
        else:
            raise SyntaxError(f'Unexpected token {token.value} at line {token.line}, column {token.col}')
    
    def eat(self, expected_type):
        if self.current_token.type != expected_type:
            raise SyntaxError(f'Expected {expected_type}, got {self.current_token.value} at line {self.current_token.line}, column {self.current_token.col}')
        self.advance()
    
    def advance(self):
        self.current_token = next(self.tokens, Token('EOF', '', 0, 0))

#Intermediate Code Generator
class ThreeAddressCodeGenerator:
    def __init__(self):
        self.instructions = []
        self.temp_counter = 0
        self.label_counter = 0
        self.string_literals = {}
        self.string_counter = 0
    
    def generate(self, ast):
        self.instructions = []
        self.temp_counter = 0
        self.label_counter = 0
        self.string_literals = {}
        self.string_counter = 0
        
        for stmt in ast['body']:
            self._generate_statement(stmt)
        
        return self.instructions
    
    def _generate_statement(self, node):
        if node['type'] == 'assignment':
            self._generate_assignment(node)
        elif node['type'] == 'print':
            self._generate_print(node)
        elif node['type'] == 'if_statement':
            self._generate_if(node)
        elif node['type'] == 'else_statement':
            self._generate_else(node)
        elif node['type'] == 'elif_statement':
            self._generate_elif(node)
        elif node['type'] == 'while_statement':
            self._generate_while(node)
        elif node['type'] == 'for_statement':
            self._generate_for(node)
        elif node['type'] == 'break_statement':
            self._generate_break(node)
        elif node['type'] == 'continue_statement':
            self._generate_continue(node)
        elif node['type'] == 'return_statement':
            self._generate_return(node)
        elif node['type'] == 'function_declaration':
            self._generate_function(node)
    
    def _generate_assignment(self, node):
        result = self._generate_expression(node['value'])
        target = node['target']
        self.instructions.append(f"{target} = {result}")
    
    def _generate_print(self, node):
        result = self._generate_expression(node['expression'])
        self.instructions.append(f"PRINT {result}")
    
    def _generate_if(self, node):
        condition_result = self._generate_expression(node['condition'])
        else_label = self._new_label()
        end_label = self._new_label()
        self.instructions.append(f"IF NOT {condition_result} GOTO {else_label}")
        for stmt in node['body']['body']:
            self._generate_statement(stmt)
        self.instructions.append(f"GOTO {end_label}")
        self.instructions.append(f"{else_label}:")
        self.instructions.append(f"{end_label}:")
    
    def _generate_else(self, node):
        for stmt in node['body']['body']:
            self._generate_statement(stmt)
    
    def _generate_elif(self, node):
        condition_result = self._generate_expression(node['condition'])
        else_label = self._new_label()
        end_label = self._new_label()
        self.instructions.append(f"IF NOT {condition_result} GOTO {else_label}")
        for stmt in node['body']['body']:
            self._generate_statement(stmt)
        self.instructions.append(f"GOTO {end_label}")
        self.instructions.append(f"{else_label}:")
    
    def _generate_while(self, node):
        loop_start = self._new_label()
        loop_end = self._new_label()
        self.instructions.append(f"{loop_start}:")
        condition_result = self._generate_expression(node['condition'])
        self.instructions.append(f"IF NOT {condition_result} GOTO {loop_end}")
        for stmt in node['body']['body']:
            self._generate_statement(stmt)
        self.instructions.append(f"GOTO {loop_start}")
        self.instructions.append(f"{loop_end}:")
    
    def _generate_for(self, node):
        init_label = self._new_label()
        loop_start = self._new_label()
        loop_end = self._new_label()
        
        # Initialization
        self.instructions.append(f"{init_label}:")
        self._generate_expression(node['init'])
        
        # Condition check
        self.instructions.append(f"{loop_start}:")
        condition_result = self._generate_expression(node['condition'])
        self.instructions.append(f"IF NOT {condition_result} GOTO {loop_end}")
        
        # Body execution
        for stmt in node['body']['body']:
            self._generate_statement(stmt)
        
        # Update
        self._generate_expression(node['update'])
        self.instructions.append(f"GOTO {loop_start}")
        
        # End of loop
        self.instructions.append(f"{loop_end}:")
    
    def _generate_break(self, node):
        # Implementation of break would require tracking loop labels
        # For simplicity, we'll just add a comment here
        self.instructions.append("BREAK")  # This would need more complex handling in a real compiler
    
    def _generate_continue(self, node):
        # Implementation of continue would require tracking loop labels
        self.instructions.append("CONTINUE")  # This would need more complex handling
    
    def _generate_return(self, node):
        result = self._generate_expression(node['expression'])
        self.instructions.append(f"RETURN {result}")
    
    def _generate_function(self, node):
        self.instructions.append(f"FUNCTION {node['name']}:")
        for stmt in node['body']['body']:
            self._generate_statement(stmt)
        self.instructions.append(f"END FUNCTION {node['name']}")
    
    def _generate_expression(self, node):
        if node['type'] == 'number':
            return str(node['value'])
        elif node['type'] == 'string':
            string_value = node['value']
            if string_value not in self.string_literals:
                string_var = f"str_{self.string_counter}"
                self.string_counter += 1
                self.string_literals[string_value] = string_var
                self.instructions.append(f"{string_var} = \"{string_value}\"")
            return self.string_literals[string_value]
        elif node['type'] == 'identifier':
            return node['name']
        elif node['type'] == 'binary_expression':
            left = self._generate_expression(node['left'])
            right = self._generate_expression(node['right'])
            operator = self._get_operator(node['operator'])
            
            temp = self._new_temp()
            
            if node['operator'] == 'EXPONENT':
                self.instructions.append(f"{temp} = CALL pow {left}, {right}")
            else:
                self.instructions.append(f"{temp} = {left} {operator} {right}")
            
            return temp
        elif node['type'] == 'unary_expression':
            operand = self._generate_expression(node['argument'])
            temp = self._new_temp()
            self.instructions.append(f"{temp} = -{operand}")
            return temp
    
    def _get_operator(self, op):
        return {
            'PLUS': '+',
            'MINUS': '-',
            'MULTIPLY': '*',
            'DIVIDE': '/',
            'MODULUS': '%',
            'EXPONENT': '^',
            'EQUAL': '==',
            'NOT_EQUAL': '!=',
            'LESS': '<',
            'LESS_EQUAL': '<=',
            'GREATER': '>',
            'GREATER_EQUAL': '>='
        }[op]
    
    def _new_temp(self):
        temp = f"t{self.temp_counter}"
        self.temp_counter += 1
        return temp
    
    def _new_label(self):
        label = f"L{self.label_counter}"
        self.label_counter += 1
        return label

#Code Generator
class TACToCCodeConverter:
    def __init__(self):
        self.c_code = []
        self.indent_level = 0
        self.declared_vars = set()
        self.temp_vars = set()
    
    def convert(self, tac_instructions):
        self.c_code = []
        self.declared_vars = set()
        self.temp_vars = set()
        
        # Generate C code header
        self.c_code.append('#include <stdio.h>')
        self.c_code.append('#include <math.h>')
        self.c_code.append('int main() {')
        self.indent_level += 1
        
        # First pass - identify all variables that need to be declared
        for instr in tac_instructions:
            if '=' in instr and not instr.strip().startswith('IF') and not ':' in instr:
                var_name = instr.split('=')[0].strip()
                if var_name.startswith('t'):
                    self.temp_vars.add(var_name)
                else:
                    self.declared_vars.add(var_name)
        
        # Declare all variables
        for var in sorted(self.declared_vars):
            c_var = self._hindi_to_c_id(var)
            self._add_line(f"double {c_var};")
        
        # Declare temporary variables
        if self.temp_vars:
            temp_vars_list = ", ".join(sorted(self.temp_vars))
            self._add_line(f"double {temp_vars_list};")
        
        # Second pass - convert TAC to C code
        skip_lines = 0
        for i, instr in enumerate(tac_instructions):
            if skip_lines > 0:
                skip_lines -= 1
                continue
                
            # Handle labels
            if instr.endswith(':'):
                label = instr[:-1]
                self._add_line(f"{label}:")
            
            # Handle conditional jumps
            elif instr.startswith('IF NOT'):
                condition = instr[7:].split('GOTO')[0].strip()
                label = instr.split('GOTO')[1].strip()
                self._add_line(f"if (!({condition})) goto {label};")
            
            # Handle unconditional jumps
            elif instr.startswith('GOTO'):
                label = instr[5:].strip()
                self._add_line(f"goto {label};")
            
            # Handle print statements
            elif instr.startswith('PRINT'):
                var = instr[6:].strip()
                c_var = var if var.startswith('t') else self._hindi_to_c_id(var)
                self._add_line(f'printf("%f\\n", {c_var});')
            
            # Handle function calls (like pow)
            elif 'CALL pow' in instr:
                args = instr.split('CALL pow')[1].strip().split(',')
                arg1 = args[0].strip()
                arg2 = args[1].strip()
                c_arg1 = arg1 if arg1.startswith('t') else self._hindi_to_c_id(arg1)
                c_arg2 = arg2 if arg2.startswith('t') else self._hindi_to_c_id(arg2)
                self._add_line(f"{c_arg1} = pow({c_arg1}, {c_arg2});")
            
            # Handle assignments
            elif '=' in instr:
                lhs, rhs = instr.split('=', 1)
                lhs = lhs.strip()
                rhs = rhs.strip()
                
                c_lhs = lhs if lhs.startswith('t') else self._hindi_to_c_id(lhs)
                
                # Replace Hindi variable names with their C equivalents
                c_rhs = rhs
                for var in self.declared_vars:
                    if var in c_rhs and not var.startswith('t'):
                        c_var = self._hindi_to_c_id(var)
                        c_rhs = re.sub(r'\b' + re.escape(var) + r'\b', c_var, c_rhs)
                
                self._add_line(f"{c_lhs} = {c_rhs};")
        
        # Add return statement and close main function
        self._add_line('return 0;')
        self.indent_level -= 1
        self.c_code.append('}')
        
        return '\n'.join(self.c_code)
    
    def _hindi_to_c_id(self, hindi_id):
        transliteration_map = {
            'अ': 'a', 'आ': 'aa', 'इ': 'i', 'ई': 'ee', 'उ': 'u', 'ऊ': 'oo',
            'ए': 'e', 'ऐ': 'ai', 'ओ': 'o', 'औ': 'au',
            'क': 'k', 'ख': 'kh', 'ग': 'g', 'घ': 'gh', 'ङ': 'ng',
            'च': 'ch', 'छ': 'chh', 'ज': 'j', 'झ': 'jh', 'ञ': 'nj',
            'ट': 't', 'ठ': 'th', 'ड': 'd', 'ढ': 'dh', 'ण': 'n',
            'त': 't', 'थ': 'th', 'द': 'd', 'ध': 'dh', 'न': 'n',
            'प': 'p', 'फ': 'ph', 'ब': 'b', 'भ': 'bh', 'म': 'm',
            'य': 'y', 'र': 'r', 'ल': 'l', 'व': 'v', 'श': 'sh',
            'ष': 'sh', 'स': 's', 'ह': 'h',
            'ँ': 'm', 'ं': 'n', 'ः': 'h', '़': '', 'ऽ': '',
            'ा': 'a', 'ि': 'i', 'ी': 'ee', 'ु': 'u', 'ू': 'oo',
            'े': 'e', 'ै': 'ai', 'ो': 'o', 'ौ': 'au',
            'ॠ': 'ri', 'ऌ': 'li', 'ॡ': 'lii', 'ॐ': 'om',
            '1': '1', '2': '2', '3': '3', '4': '4', '5': '5',
            '6': '6', '7': '7', '8': '8', '9': '9', '0': '0'
        }
        
        c_id = []
        for char in hindi_id:
            c_id.append(transliteration_map.get(char, '_'))
        return 'var_' + ''.join(c_id)
    
    def _add_line(self, code):
        self.c_code.append(f'{"  " * self.indent_level}{code}')

class HindiCompiler:
    def __init__(self, source_code):
        self.source_code = source_code
        self.ast = None
        self.tac_code = None
        self.c_code = None
    
    def generate_ast(self):
        lexer = HindiLexer(self.source_code)
        tokens = lexer.tokenize()
        parser = HindiParser(tokens)
        self.ast = parser.parse()
        return self.ast
    
    def generate_tac(self):
        if not self.ast:
            self.generate_ast()
        
        tac_generator = ThreeAddressCodeGenerator()
        self.tac_code = tac_generator.generate(self.ast)
        return self.tac_code
    
    def generate_c_code(self, output_file='output.c'):
        if not self.tac_code:
            self.generate_tac()
        
        tac_to_c = TACToCCodeConverter()
        self.c_code = tac_to_c.convert(self.tac_code)
        
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(self.c_code)
            full_path = os.path.abspath(output_file)
            print(f"\n✅ C code generated: {full_path}")
            print("📄 Generated C code preview:")
            print(self.c_code[:200] + ("..." if len(self.c_code) > 200 else ""))
            return self.c_code
        except Exception as e:
            print(f"❌ Error writing C code: {str(e)}")
            return None
    
    def save_tac(self, output_file='output.tac'):
        """Save the generated three-address code to a file."""
        if not self.tac_code:
            self.generate_tac()
        
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                for instr in self.tac_code:
                    f.write(f"{instr}\n")
            full_path = os.path.abspath(output_file)
            print(f"\n✅ Three-address code generated: {full_path}")
            print("📄 Generated TAC preview:")
            for i, instr in enumerate(self.tac_code[:10]):
                print(f"{i+1}: {instr}")
            if len(self.tac_code) > 10:
                print("...")
            return True
        except Exception as e:
            print(f"❌ Error writing TAC: {str(e)}")
            return False
    
    def compile_and_run(self, output_name='program'):
        if not self.c_code:
            self.generate_c_code(f'{output_name}.c')
        
        try:
            # Compile the C code
            compile_result = subprocess.run([
                'gcc',
                f'{output_name}.c',
                '-o', output_name,
                '-lm'
            ], capture_output=True, text=True)
            
            if compile_result.returncode != 0:
                print(f"\n❌ Compilation failed:")
                print(compile_result.stderr)
                return False
            
            # Run the compiled program
            run_result = subprocess.run([f'./{output_name}'], capture_output=True, text=True)
            
            print(f"\n📤 Program Output:")
            if run_result.stdout:
                print(run_result.stdout)
            if run_result.stderr:
                print(f"Error: {run_result.stderr}")
            
            return run_result.returncode == 0
        except FileNotFoundError:
            print("❌ Error: Install gcc compiler first")
            return False

# Test Code
if __name__ == "__main__":
    test_code = """
    संख्या1 = 10;
    संख्या2 = 20;
    योग = संख्या1 + संख्या2;
    प्रिंट(योग);
    """
    
    print("🧪 Testing Hindi compiler with three-address code generation...")
    compiler = HindiCompiler(test_code)
    
    print("\n🔍 Generating AST...")
    ast = compiler.generate_ast()
    
    print("\n📝 Generating Three-Address Code...")
    tac = compiler.generate_tac()
    compiler.save_tac("test_output.tac")
    
    print("\n🔧 Generating C code from Three-Address Code...")
    c_code = compiler.generate_c_code('test_output.c')
    
    if c_code:
        compiler.compile_and_run('test_output')