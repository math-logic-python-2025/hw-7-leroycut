# This file is part of the materials accompanying the book
# "Mathematical Logic through Python" by Gonczarowski and Nisan,
# Cambridge University Press. Book site: www.LogicThruPython.org
# (c) Yannai A. Gonczarowski and Noam Nisan, 2017-2022
# File name: predicates/syntax.py

"""Syntactic handling of predicate-logic expressions."""

from __future__ import annotations
from functools import lru_cache
from typing import AbstractSet, Mapping, Optional, Sequence, Set, Tuple, Union

from src.logic_utils import (
    frozen,
    memoized_parameterless_method,
)

from src.propositions.syntax import (
    Formula as PropositionalFormula,
)


class ForbiddenVariableError(Exception):
    """Raised by `Term.substitute` and `Formula.substitute` when a substituted
    term contains a variable name that is forbidden in that context.

    Attributes:
        variable_name (`str`): the variable name that was forbidden in the
            context in which a term containing it was to be substituted.
    """

    variable_name: str

    def __init__(self, variable_name: str):
        """Initializes a `ForbiddenVariableError` from the offending variable
        name.

        Parameters:
            variable_name: variable name that is forbidden in the context in
                which a term containing it is to be substituted.
        """
        assert is_variable(variable_name)
        self.variable_name = variable_name


@lru_cache(maxsize=100)  # Cache the return value of is_constant
def is_constant(string: str) -> bool:
    """Checks if the given string is a constant name.

    Parameters:
        string: string to check.

    Returns:
        ``True`` if the given string is a constant name, ``False`` otherwise.
    """
    return (
        ((string[0] >= "0" and string[0] <= "9") or (string[0] >= "a" and string[0] <= "e")) and string.isalnum()
    ) or string == "_"


@lru_cache(maxsize=100)  # Cache the return value of is_variable
def is_variable(string: str) -> bool:
    """Checks if the given string is a variable name.

    Parameters:
        string: string to check.

    Returns:
        ``True`` if the given string is a variable name, ``False`` otherwise.
    """
    return string[0] >= "u" and string[0] <= "z" and string.isalnum()


@lru_cache(maxsize=100)  # Cache the return value of is_function
def is_function(string: str) -> bool:
    """Checks if the given string is a function name.

    Parameters:
        string: string to check.

    Returns:
        ``True`` if the given string is a function name, ``False`` otherwise.
    """
    return string[0] >= "f" and string[0] <= "t" and string.isalnum()


@frozen
class Term:
    """An immutable predicate-logic term in tree representation, composed from
    variable names and constant names, and function names applied to them.

    Attributes:
        root (`str`): the constant name, variable name, or function name at the
            root of the term tree.
        arguments (`~typing.Optional`\\[`~typing.Tuple`\\[`Term`, ...]]): the
            arguments of the root, if the root is a function name.
    """

    root: str
    arguments: Optional[Tuple[Term, ...]]

    def __init__(self, root: str, arguments: Optional[Sequence[Term]] = None):
        """Initializes a `Term` from its root and root arguments.

        Parameters:
            root: the root for the formula tree.
            arguments: the arguments for the root, if the root is a function
                name.
        """
        if is_constant(root) or is_variable(root):
            assert arguments is None
            self.root = root
        else:
            assert is_function(root)
            assert arguments is not None and len(arguments) > 0
            self.root = root
            self.arguments = tuple(arguments)

    @memoized_parameterless_method
    def __repr__(self) -> str:
        """Computes the string representation of the current term.

        Returns:
            The standard string representation of the current term.
        """
        # Task 7.1
        def get_S():
            S = f'{root}('
            for i in range(len(self.arguments)):
                S += f'{self.arguments[i]}'
                if i != len(self.arguments) - 1:
                    S += ','
            S += ')'
            return S
        
        root = self.root
        if is_constant(root) or is_variable(root):
            return root
        
        return get_S()

    def __eq__(self, other: object) -> bool:
        """Compares the current term with the given one.

        Parameters:
            other: object to compare to.

        Returns:
            ``True`` if the given object is a `Term` object that equals the
            current term, ``False`` otherwise.
        """
        return isinstance(other, Term) and str(self) == str(other)

    def __ne__(self, other: object) -> bool:
        """Compares the current term with the given one.

        Parameters:
            other: object to compare to.

        Returns:
            ``True`` if the given object is not a `Term` object or does not
            equal the current term, ``False`` otherwise.
        """
        return not self == other

    def __hash__(self) -> int:
        return hash(str(self))

    @staticmethod
    def _parse_prefix(string: str) -> Tuple[Term, str]:
        """Parses a prefix of the given string into a term.

        Parameters:
            string: string to parse, which has a prefix that is a valid
                representation of a term.

        Returns:
            A pair of the parsed term and the unparsed suffix of the string. If
            the given string has as a prefix a constant name (e.g., ``'c12'``)
            or a variable name (e.g., ``'x12'``), then the parsed prefix will be
            that entire name (and not just a part of it, such as ``'x1'``).
        """
        # Task 7.3a
        
        def split_by_bracket(s: str):
            first_pairs = {'[': ']', '(': ')'}
            first_char = s[0]
            last_char = first_pairs[first_char]
            count = 1
            
            for i, char in enumerate(s[1:], 1):
                if char == first_char:
                    count += 1
                elif char == last_char:
                    count -= 1
                    if count == 0:
                        return s[:i+1], s[i+1:]

        def parse_only_num(s: str):
            idx = 0
            while idx < len(s) and s[idx].isalnum():
                idx += 1
            return s[:idx], s[idx:]

        def parse_one_char(s: str):
            if s[0] == '_':
                return '_', s[1:]
            char, rest = parse_only_num(s)
            if is_constant(s[0]) or is_variable(s[0]):
                return char, rest
            return None, s

        first_char = string[0]
        
        if is_constant(first_char) or is_variable(first_char):
            char, rest = parse_one_char(string)
            return Term(char), rest
        
        num, rest = parse_only_num(string)
        args_str, suf = split_by_bracket(rest)
        
        remaining_string = args_str[1:-1]
        parsed_args = []
        
        while remaining_string:
            arg, remaining_string = parse_one_char(remaining_string)
            if arg is None:
                arg_term, remaining_string = Term._parse_prefix(remaining_string)
                parsed_args.append(arg_term)
            else:
                parsed_args.append(Term(arg))
            
            if remaining_string.startswith(','):
                remaining_string = remaining_string[1:]
        
        return Term(num, parsed_args), suf

    @staticmethod
    def parse(string: str) -> Term:
        """Parses the given valid string representation into a term.

        Parameters:
            string: string to parse.

        Returns:
            A term whose standard string representation is the given string.
        """
        # Task 7.3b
        
        return Term._parse_prefix(string)[0]

    def constants(self) -> Set[str]:
        """Finds all constant names in the current term.

        Returns:
            A set of all constant names used in the current term.
        """
        # Task 7.5a
        
        root = self.root
        if is_constant(root):
            return {root}

        elif is_variable(root):
            return set()

        elif is_function(root):
            constants = set()
            for argument in self.arguments:
                constants = constants | Term.constants(argument)
            return constants

    def variables(self) -> Set[str]:
        """Finds all variable names in the current term.

        Returns:
            A set of all variable names used in the current term.
        """
        # Task 7.5b
        
        root = self.root
        if is_constant(root):
            return set()
        
        if is_variable(root):
            return {root}

        if is_function(root):
            variables = set()
            for argument in self.arguments:
                variables = variables | Term.variables(argument)
            return variables

    def functions(self) -> Set[Tuple[str, int]]:
        """Finds all function names in the current term, along with their
        arities.

        Returns:
            A set of pairs of function name and arity (number of arguments) for
            all function names used in the current term.
        """
        # Task 7.5c

        root = self.root
        functions = set()
        if is_function(root):
            functions = {(root, len(self.arguments))}
            for argument in self.arguments:
                functions = functions | Term.functions(argument)
        return functions
        
    def substitute(
        self,
        substitution_map: Mapping[str, Term],
        forbidden_variables: AbstractSet[str] = frozenset(),
    ) -> Term:
        """Substitutes in the current term, each constant name `construct` or
        variable name `construct` that is a key in `substitution_map` with the
        term `substitution_map`\ ``[``\ `construct`\ ``]``.

        Parameters:
            substitution_map: mapping defining the substitutions to be
                performed.
            forbidden_variables: variable names not allowed in substitution
                terms.

        Returns:
            The term resulting from performing all substitutions. Only
            constant name and variable name occurrences originating in the
            current term are substituted (i.e., those originating in one of the
            specified substitutions are not subjected to additional
            substitutions).

        Raises:
            ForbiddenVariableError: If a term that is used in the requested
                substitution contains a variable name from
                `forbidden_variables`.

        Examples:
            >>> Term.parse('f(x,c)').substitute(
            ...     {'c': Term.parse('plus(d,x)'), 'x': Term.parse('c')}, {'y'})
            f(c,plus(d,x))

            >>> Term.parse('f(x,c)').substitute(
            ...     {'c': Term.parse('plus(d,y)')}, {'y'})
            Traceback (most recent call last):
              ...
            predicates.syntax.ForbiddenVariableError: y
        """
        for construct in substitution_map:
            assert is_constant(construct) or is_variable(construct)
        for variable in forbidden_variables:
            assert is_variable(variable)
        # Task 9.1


@lru_cache(maxsize=100)  # Cache the return value of is_equality
def is_equality(string: str) -> bool:
    """Checks if the given string is the equality relation.

    Parameters:
        string: string to check.

    Returns:
        ``True`` if the given string is the equality relation, ``False``
        otherwise.
    """
    return string == "="


@lru_cache(maxsize=100)  # Cache the return value of is_relation
def is_relation(string: str) -> bool:
    """Checks if the given string is a relation name.

    Parameters:
        string: string to check.

    Returns:
        ``True`` if the given string is a relation name, ``False`` otherwise.
    """
    return string[0] >= "F" and string[0] <= "T" and string.isalnum()


@lru_cache(maxsize=100)  # Cache the return value of is_unary
def is_unary(string: str) -> bool:
    """Checks if the given string is a unary operator.

    Parameters:
        string: string to check.

    Returns:
        ``True`` if the given string is a unary operator, ``False`` otherwise.
    """
    return string == "~"


@lru_cache(maxsize=100)  # Cache the return value of is_binary
def is_binary(string: str) -> bool:
    """Checks if the given string is a binary operator.

    Parameters:
        string: string to check.

    Returns:
        ``True`` if the given string is a binary operator, ``False`` otherwise.
    """
    return string == "&" or string == "|" or string == "->"


@lru_cache(maxsize=100)  # Cache the return value of is_quantifier
def is_quantifier(string: str) -> bool:
    """Checks if the given string is a quantifier.

    Parameters:
        string: string to check.

    Returns:
        ``True`` if the given string is a quantifier, ``False`` otherwise.
    """
    return string == "A" or string == "E"


@frozen
class Formula:
    """An immutable predicate-logic formula in tree representation, composed
    from relation names applied to predicate-logic terms, and operators and
    quantifications applied to them.

    Attributes:
        root (`str`): the relation name, equality relation, operator, or
            quantifier at the root of the formula tree.
        arguments (`~typing.Optional`\\[`~typing.Tuple`\\[`Term`, ...]]): the
            arguments of the root, if the root is a relation name or the
            equality relation.
        first (`~typing.Optional`\\[`Formula`]): the first operand of the root,
            if the root is a unary or binary operator.
        second (`~typing.Optional`\\[`Formula`]): the second operand of the
            root, if the root is a binary operator.
        variable (`~typing.Optional`\\[`str`]): the variable name quantified by
            the root, if the root is a quantification.
        statement (`~typing.Optional`\\[`Formula`]): the statement quantified by
            the root, if the root is a quantification.
    """

    root: str
    arguments: Optional[Tuple[Term, ...]]
    first: Optional[Formula]
    second: Optional[Formula]
    variable: Optional[str]
    statement: Optional[Formula]

    def __init__(
        self,
        root: str,
        arguments_or_first_or_variable: Union[Sequence[Term], Formula, str],
        second_or_statement: Optional[Formula] = None,
    ):
        """Initializes a `Formula` from its root and root arguments, root
        operands, or root quantified variable name and statement.

        Parameters:
            root: the root for the formula tree.
            arguments_or_first_or_variable: the arguments for the root, if the
                root is a relation name or the equality relation; the first
                operand for the root, if the root is a unary or binary operator;
                the variable name to be quantified by the root, if the root is a
                quantification.
            second_or_statement: the second operand for the root, if the root is
                a binary operator; the statement to be quantified by the root,
                if the root is a quantification.
        """
        if is_equality(root) or is_relation(root):
            # Populate self.root and self.arguments
            assert isinstance(arguments_or_first_or_variable, Sequence) and not isinstance(
                arguments_or_first_or_variable, str
            )
            if is_equality(root):
                assert len(arguments_or_first_or_variable) == 2
            assert second_or_statement is None
            self.root, self.arguments = root, tuple(arguments_or_first_or_variable)
        elif is_unary(root):
            # Populate self.first
            assert isinstance(arguments_or_first_or_variable, Formula)
            assert second_or_statement is None
            self.root, self.first = root, arguments_or_first_or_variable
        elif is_binary(root):
            # Populate self.first and self.second
            assert isinstance(arguments_or_first_or_variable, Formula)
            assert second_or_statement is not None
            self.root, self.first, self.second = (
                root,
                arguments_or_first_or_variable,
                second_or_statement,
            )
        else:
            assert is_quantifier(root)
            # Populate self.variable and self.statement
            assert isinstance(arguments_or_first_or_variable, str) and is_variable(arguments_or_first_or_variable)
            assert second_or_statement is not None
            self.root, self.variable, self.statement = (
                root,
                arguments_or_first_or_variable,
                second_or_statement,
            )

    @memoized_parameterless_method
    def __repr__(self) -> str:
        """Computes the string representation of the current formula.

        Returns:
            The standard string representation of the current formula.
        """
        # Task 7.2
        
        def get_S():
            S = f'{root}('
            for i in range(len(self.arguments)):
                S += f'{self.arguments[i]}'
                if i != len(self.arguments) - 1:
                    S += ','
            S += ')'
            return S
        
        root = self.root
        if is_unary(root):
            return f'{root}{self.first.__repr__()}'
        
        if is_binary(root):
            return f'({self.first.__repr__()}{root}{self.second.__repr__()})'
        
        if is_equality(root):
            return f'{self.arguments[0].__repr__()}{root}{self.arguments[1].__repr__()}'

        if is_relation(root):
            return get_S()
        
        return f'{root}{self.variable}[{self.statement.__repr__()}]'

    def __eq__(self, other: object) -> bool:
        """Compares the current formula with the given one.

        Parameters:
            other: object to compare to.

        Returns:
            ``True`` if the given object is a `Formula` object that equals the
            current formula, ``False`` otherwise.
        """
        return isinstance(other, Formula) and str(self) == str(other)

    def __ne__(self, other: object) -> bool:
        """Compares the current formula with the given one.

        Parameters:
            other: object to compare to.

        Returns:
            ``True`` if the given object is not a `Formula` object or does not
            equal the current formula, ``False`` otherwise.
        """
        return not self == other

    def __hash__(self) -> int:
        return hash(str(self))

    @staticmethod
    def _parse_prefix(string: str) -> Tuple[Formula, str]:
        """Parses a prefix of the given string into a formula.

        Parameters:
            string: string to parse, which has a prefix that is a valid
                representation of a formula.

        Returns:
            A pair of the parsed formula and the unparsed suffix of the string.
            If the given string has as a prefix a term followed by an equality
            followed by a constant name (e.g., ``'f(y)=c12'``) or by a variable
            name (e.g., ``'f(y)=x12'``), then the parsed prefix will include
            that entire name (and not just a part of it, such as ``'f(y)=x1'``).
        """
        # Task 7.4a
        
        def split_by_bracket(s: str):
            first_pairs = {'[': ']', '(': ')'}
            first_char = s[0]
            last_char = first_pairs[first_char]
            count = 1
            
            for i, char in enumerate(s[1:], 1):
                if char == s[0]:
                    count += 1
                elif char == last_char:
                    count -= 1
                    if count == 0:
                        return s[:i+1], s[i+1:]
        
        def parse_only_num(s: str) -> Tuple[str, str]:
            if s[0] == '_':
                return '_', s[1:]
            idx = 0
            while idx < len(s) and s[idx].isalnum():
                idx += 1
            
            if is_constant(s[0]) or is_variable(s[0]):
                return s[:idx], s[idx:]
            
            args, rest = split_by_bracket(s[idx:])
            return s[:idx] + args, rest

        def parse_operation(s: str) -> Tuple[str, str]:
            if s.startswith('<->'):
                return '<->', s[3:]
            if s[0] == '-' and len(s) > 1 and (s[1] in {'>', '&', '|'}):
                return s[:2], s[2:]
            return s[0], s[1:]

        first_char = string[0]
        if is_constant(first_char) or is_variable(first_char) or is_function(first_char):
            left, rest = parse_only_num(string)
            right, suf = parse_only_num(rest[1:])
            return Formula('=', [Term.parse(left), Term.parse(right)]), suf

        if is_relation(first_char):
            end = 0
            while end < len(string) and string[end].isalnum():
                end += 1
            name = string[:end]
            
            args_str, suf = split_by_bracket(string[end:])
            args = []
            remaining = args_str[1:-1]
            
            while remaining:
                arg, remaining = parse_only_num(remaining)
                args.append(Term._parse_prefix(arg)[0])
                if remaining and remaining[0] == ',':
                    remaining = remaining[1:]
                    
            return Formula(name, args), suf
        
        if is_unary(first_char):
            formula, suf = Formula._parse_prefix(string[1:])
            return Formula('~', formula), suf

        if is_quantifier(first_char):
            num, rest = parse_only_num(string[1:])
            state, suf = split_by_bracket(rest)
            in_formula = Formula._parse_prefix(state[1:-1])[0]
            return Formula(first_char, num, in_formula), suf

        if first_char == '(':
            brackect_str, suf = split_by_bracket(string)
            left, rest = Formula._parse_prefix(brackect_str[1:])
            op, remaining = parse_operation(rest)
            right, _ = Formula._parse_prefix(remaining)
            return Formula(op, left, right), suf

    @staticmethod
    def parse(string: str) -> Formula:
        """Parses the given valid string representation into a formula.

        Parameters:
            string: string to parse.

        Returns:
            A formula whose standard string representation is the given string.
        """
        # Task 7.4b
        
        return Formula._parse_prefix(string)[0]
        
    def constants(self) -> Set[str]:
        """Finds all constant names in the current formula.

        Returns:
            A set of all constant names used in the current formula.
        """
        # Task 7.6a

        root = self.root
        if is_unary(root):
            return Formula.constants(self.first)

        if is_binary(root):
            return Formula.constants(self.first) | Formula.constants(self.second)

        if is_quantifier(root):
            return Formula.constants(self.statement)
        
        if is_equality(root) or is_relation(root):
            constants = set()
            for argument in self.arguments:
                constants = constants | Term.constants(argument)
            return constants

    def variables(self) -> Set[str]:
        """Finds all variable names in the current formula.

        Returns:
            A set of all variable names used in the current formula.
        """
        # Task 7.6b
        
        root = self.root
        if is_unary(root):
            return Formula.variables(self.first)

        if is_binary(root):
            return Formula.variables(self.first) | Formula.variables(self.second)

        if is_quantifier(root):
            return Formula.variables(self.statement) | {self.variable}

        if is_equality(root) or is_relation(root):
            variables = set()
            for argument in self.arguments:
                variables = variables | Term.variables(argument)
            return variables

    def free_variables(self) -> Set[str]:
        """Finds all variable names that are free in the current formula.

        Returns:
            A set of every variable name that is used in the current formula not
            only within a scope of a quantification on that variable name.
        """
        # Task 7.6c
        
        root = self.root
        if is_unary(root):
            return Formula.free_variables(self.first)

        if is_binary(root):
            return Formula.free_variables(self.first) | Formula.free_variables(self.second)

        if is_quantifier(root):
            return Formula.free_variables(self.statement) - {self.variable}
        
        if is_equality(root) or is_relation(root):
            free_variables = set()
            for argument in self.arguments:
                free_variables = free_variables | Term.variables(argument)
            return free_variables

    def functions(self) -> Set[Tuple[str, int]]:
        """Finds all function names in the current formula, along with their
        arities.

        Returns:
            A set of pairs of function name and arity (number of arguments) for
            all function names used in the current formula.
        """
        # Task 7.6d
        
        root = self.root
        if is_unary(root):
            return Formula.functions(self.first)

        if is_binary(root):
            return Formula.functions(self.first) | Formula.functions(self.second)

        if is_quantifier(root):
            return Formula.functions(self.statement)

        if is_equality(root) or is_relation(root):
            functions = set()
            for argument in self.arguments:
                functions = functions | Term.functions(argument)
            return functions

    def relations(self) -> Set[Tuple[str, int]]:
        """Finds all relation names in the current formula, along with their
        arities.

        Returns:
            A set of pairs of relation name and arity (number of arguments) for
            all relation names used in the current formula.
        """
        # Task 7.6e
        
        root = self.root
        if is_unary(root):
            return Formula.relations(self.first)

        if is_binary(root):
            return Formula.relations(self.first) | Formula.relations(self.second)

        if is_quantifier(root):
            return Formula.relations(self.statement)
        
        if is_relation(root):
            return {(root, len(self.arguments))}
        
        return set()

    def substitute(
        self,
        substitution_map: Mapping[str, Term],
        forbidden_variables: AbstractSet[str] = frozenset(),
    ) -> Formula:
        """Substitutes in the current formula, each constant name `construct` or
        free occurrence of variable name `construct` that is a key in
        `substitution_map` with the term
        `substitution_map`\ ``[``\ `construct`\ ``]``.

        Parameters:
            substitution_map: mapping defining the substitutions to be
                performed.
            forbidden_variables: variable names not allowed in substitution
                terms.

        Returns:
            The formula resulting from performing all substitutions. Only
            constant name and variable name occurrences originating in the
            current formula are substituted (i.e., those originating in one of
            the specified substitutions are not subjected to additional
            substitutions).

        Raises:
            ForbiddenVariableError: If a term that is used in the requested
                substitution contains a variable name from `forbidden_variables`
                or a variable name occurrence that becomes bound when that term
                is substituted into the current formula.

        Examples:
            >>> Formula.parse('Ay[x=c]').substitute(
            ...     {'c': Term.parse('plus(d,x)'), 'x': Term.parse('c')}, {'z'})
            Ay[c=plus(d,x)]

            >>> Formula.parse('Ay[x=c]').substitute(
            ...     {'c': Term.parse('plus(d,z)')}, {'z'})
            Traceback (most recent call last):
              ...
            predicates.syntax.ForbiddenVariableError: z

            >>> Formula.parse('Ay[x=c]').substitute(
            ...     {'c': Term.parse('plus(d,y)')})
            Traceback (most recent call last):
              ...
            predicates.syntax.ForbiddenVariableError: y
        """
        for construct in substitution_map:
            assert is_constant(construct) or is_variable(construct)
        for variable in forbidden_variables:
            assert is_variable(variable)
        # Task 9.2

    def propositional_skeleton(
        self,
    ) -> Tuple[PropositionalFormula, Mapping[str, Formula]]:
        """Computes a propositional skeleton of the current formula.

        Returns:
            A pair. The first element of the pair is a propositional formula
            obtained from the current formula by substituting every (outermost)
            subformula that has a relation name, equality, or quantifier at its
            root with a propositional variable name, consistently such that
            multiple identical such (outermost) subformulas are substituted with
            the same propositional variable name. The propositional variable
            names used for substitution are obtained, from left to right
            (considering their first occurrence), by calling
            `next`\ ``(``\ `~logic_utils.fresh_variable_name_generator`\ ``)``.
            The second element of the pair is a mapping from each propositional
            variable name to the subformula for which it was substituted.

        Examples:
            >>> formula = Formula.parse('((Ax[x=7]&x=7)|(~Q(y)->x=7))')
            >>> formula.propositional_skeleton()
            (((z1&z2)|(~z3->z2)), {'z1': Ax[x=7], 'z2': x=7, 'z3': Q(y)})
            >>> formula.propositional_skeleton()
            (((z4&z5)|(~z6->z5)), {'z4': Ax[x=7], 'z5': x=7, 'z6': Q(y)})
        """
        # Task 9.8

    @staticmethod
    def from_propositional_skeleton(skeleton: PropositionalFormula, substitution_map: Mapping[str, Formula]) -> Formula:
        """Computes a predicate-logic formula from a propositional skeleton and
        a substitution map.

        Arguments:
            skeleton: propositional skeleton for the formula to compute,
                containing no constants or operators beyond ``'~'``, ``'->'``,
                ``'|'``, and ``'&'``.
            substitution_map: mapping from each propositional variable name of
                the given propositional skeleton to a predicate-logic formula.

        Returns:
            A predicate-logic formula obtained from the given propositional
            skeleton by substituting each propositional variable name with the
            formula mapped to it by the given map.

        Examples:
            >>> Formula.from_propositional_skeleton(
            ...     PropositionalFormula.parse('((z1&z2)|(~z3->z2))'),
            ...     {'z1': Formula.parse('Ax[x=7]'), 'z2': Formula.parse('x=7'),
            ...      'z3': Formula.parse('Q(y)')})
            ((Ax[x=7]&x=7)|(~Q(y)->x=7))

            >>> Formula.from_propositional_skeleton(
            ...     PropositionalFormula.parse('((z9&z2)|(~z3->z2))'),
            ...     {'z2': Formula.parse('x=7'), 'z3': Formula.parse('Q(y)'),
            ...      'z9': Formula.parse('Ax[x=7]')})
            ((Ax[x=7]&x=7)|(~Q(y)->x=7))
        """
        for operator in skeleton.operators():
            assert is_unary(operator) or is_binary(operator)
        for variable in skeleton.variables():
            assert variable in substitution_map
        # Task 9.10
