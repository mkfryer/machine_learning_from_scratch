# specs.py
"""Unit tests for Nearest Neigbor.
<Michael Fryer>
<Class>
<Date>
"""
import numpy as np
import nearest_neighbor as nn
import math

# def test_exhaustive_search():
#     A = np.array([[3,7], [,7]])
#     z = np.array([1,1])
#     print(nn.exhaustive_search(A, z))
#     assert nn.exhaustive_search(A, z) == np.array([1,5,5]), 4


# def test_add():
#     assert specs.add(1, 3) == 4, "failed on positive integers"
#     assert specs.add(-5, -7) == -12, "failed on negative integers"
#     assert specs.add(-6, 14) == 8

# def test_divide():
#     assert specs.divide(4,2) == 2, "integer division"
#     assert specs.divide(5,4) == 1.25, "float division"
#     with pytest.raises(ZeroDivisionError) as excinfo:
#         specs.divide(4, 0)
#     assert excinfo.value.args[0] == "second input cannot be zero"

# # Problem 1: write a unit test for specs.smallest_factor(), then correct it.
# def test_smallest_factor():
#     assert specs.smallest_factor(6) == 2, "find smallest factor of 6"
#     assert specs.smallest_factor(9) == 3, "smallest factor is square root"
#     assert specs.smallest_factor(3) == 3, "smallest factor is second smallest factor"
#     assert specs.smallest_factor(2) == 2, "smallest factor is the smallest factor"


# # Problem 2: write a unit test for specs.month_length().
# def test_month_length():
#     assert specs.month_length("January") == 31, "days of jan"
#     assert specs.month_length("February") == 28, "days of feb"
#     assert specs.month_length("March") == 31, "days of march"
#     assert specs.month_length("April") == 30, "days of apr"
#     assert specs.month_length("May") == 31, "days of may"
#     assert specs.month_length("June") == 30, "days of june"
#     assert specs.month_length("July") == 31, "days of july"
#     assert specs.month_length("August") == 31, "days of aug"
#     assert specs.month_length("September") == 30, "days of sep"
#     assert specs.month_length("October") == 31, "days of oct"
#     assert specs.month_length("November") == 30, "days of nov"
#     assert specs.month_length("January") == 31, "days of dec"
#     assert specs.month_length("February", True) == 29, "days of feb duuring leap year"
#     assert specs.month_length("nope", True) is None, "Invalid month"
# # Problem 3: write a unit test for specs.operate().

# def test_operate():
#     assert specs.operate(2,2, "+") == 4, "addition of two positives"
#     assert specs.operate(-2,-2, "+") == -4, "addition of two negatives"
#     assert specs.operate(2,-2, "+") == 0, "addition of pos and neg"
    
#     assert specs.operate(2,2, "-") == 0, "subtraction of two positives"
#     assert specs.operate(-2,-2, "-") == 0, "subtraction of two negatives"
#     assert specs.operate(2,-2, "-") == 4, "subtraction of pos and neg"
    
#     assert specs.operate(2,2, "*") == 4, "multplication of two positives"
#     assert specs.operate(-2,-2, "*") == 4, "multplication of two negatives"
#     assert specs.operate(2,-2, "*") == -4, "multplication of pos and neg"
#     assert specs.operate(2,0, "*") == 0, "multplication of zero"

#     assert specs.operate(2,2, "/") == 1, "division of two positives"
#     assert specs.operate(-2,-2, "/") == 1, "division of two negatives"
#     assert specs.operate(2,-2, "/") == -1, "division of pos and neg"

#     with pytest.raises(TypeError) as excinfo:
#         specs.operate(1,1,13)
#     assert excinfo.value.args[0] == "oper must be a string"

#     with pytest.raises(ValueError) as excinfo:
#         specs.operate(1,4,"%")
#     assert excinfo.value.args[0] == "oper must be one of '+', '/', '-', or '*'"

#     with pytest.raises(ZeroDivisionError) as excinfo:
#         specs.operate(0,0,"/")
#     assert excinfo.value.args[0] == "division by zero is undefined"

# # Problem 4: write unit tests for specs.Fraction, then correct it.
# @pytest.fixture
# def set_up_fractions():
#     frac_1_3 = specs.Fraction(1, 3)
#     frac_1_2 = specs.Fraction(1, 2)
#     frac_n2_3 = specs.Fraction(-2, 3)
#     return frac_1_3, frac_1_2, frac_n2_3

# def test_fraction_init(set_up_fractions):
#     frac_1_3, frac_1_2, frac_n2_3 = set_up_fractions
#     assert frac_1_3.numer == 1
#     assert frac_1_2.denom == 2
#     assert frac_n2_3.numer == -2
#     frac = specs.Fraction(30, 42)
#     assert frac.numer == 5
#     assert frac.denom == 7

# def test_fraction_str(set_up_fractions):
#     frac_1_3, frac_1_2, frac_n2_3 = set_up_fractions
#     assert str(frac_1_3) == "1/3"
#     assert str(frac_1_2) == "1/2"
#     assert str(frac_n2_3) == "-2/3"

# def test_fraction_float(set_up_fractions):
#     frac_1_3, frac_1_2, frac_n2_3 = set_up_fractions
#     assert float(frac_1_3) == 1 / 3.
#     assert float(frac_1_2) == .5
#     assert float(frac_n2_3) == -2 / 3.

# def test_fraction_eq(set_up_fractions):
#     frac_1_3, frac_1_2, frac_n2_3 = set_up_fractions
#     assert frac_1_2 == specs.Fraction(1, 2)
#     assert frac_1_3 == specs.Fraction(2, 6)
#     assert frac_n2_3 == specs.Fraction(8, -12)

# def test_init(set_up_fractions):
#     with pytest.raises(ZeroDivisionError) as excinfo:
#         specs.Fraction(1,0)
#     assert excinfo.value.args[0] == "denominator cannot be zero"
    
#     with pytest.raises(TypeError) as excinfo:
#         specs.Fraction(1.4, 2.2)
#     assert excinfo.value.args[0] == "numerator and denominator must be integers"

# def test__str__(set_up_fractions):
#     frac_1_3, frac_1_2, frac_n2_3 = set_up_fractions
#     assert str(frac_1_2) == "1/2"
#     assert str(frac_1_3) == "1/3"
#     assert str(specs.Fraction(2, 1)) == "2"

# def test__float__(set_up_fractions):
#     frac_1_3, frac_1_2, frac_n2_3 = set_up_fractions
#     assert str(frac_1_2) == "1/2"
#     assert str(frac_1_3) == "1/3"
#     assert str(frac_n2_3) == "-2/3"
    
# def test__equal__(set_up_fractions):
#     frac_1_2_1 = specs.Fraction(1,2)
#     frac_1_2_2 = specs.Fraction(1,2)
#     frac_1_3 = specs.Fraction(1,3)

#     assert frac_1_2_1 == frac_1_2_2
#     assert frac_1_3 != frac_1_2_2
#     assert frac_1_2_1 == 1/2
#     assert frac_1_2_1 == .5
#     assert frac_1_2_1 != 1/3

# def test___add__(set_up_fractions):
#     frac_1_3, frac_1_2, frac_n2_3 = set_up_fractions
#     frac_5_6 = specs.Fraction(5, 6)
#     frac_n1_3 = specs.Fraction(-1, 3)

#     assert frac_1_3 + frac_1_2 == frac_5_6
#     assert frac_1_3 + frac_n1_3 == 0

# def test___sub__(set_up_fractions):
#     frac_1_3, frac_1_2, frac_n2_3 = set_up_fractions
#     frac_2_3 = specs.Fraction(2, 3)
#     frac_n1_3 = specs.Fraction(-1, 3)
#     frac_1_6 = specs.Fraction(1, 6)

#     assert frac_1_3 - frac_n1_3 == frac_2_3
#     assert frac_1_2 - frac_1_3 == frac_1_6

# def test__mul__(set_up_fractions):
#     frac_1_6 = specs.Fraction(1, 6)
#     frac_n2_9 = specs.Fraction(-2, 9)

#     frac_1_3, frac_1_2, frac_n2_3 = set_up_fractions
#     assert frac_1_3 * frac_1_2 == frac_1_6
#     assert frac_1_3 * frac_n2_3 == frac_n2_9

# def test__truediv__(set_up_fractions):
#     frac_1_3, frac_1_2, frac_n2_3 = set_up_fractions
#     frac_2_3 = specs.Fraction(2,3)
#     frac_0_1 = specs.Fraction(0,1)

#     assert frac_1_3 / frac_1_2 == frac_2_3

#     with pytest.raises(ZeroDivisionError) as excinfo:
#         frac_1_3 / frac_0_1
#     assert excinfo.value.args[0] == "cannot divide by zero"
    
# def test_count_sets():
#     small_hand = ["1111"]
#     duplicate_hand = ["1022", "1022", "1022", "0022", "1111", "1112", "1022", "1022", "1022", "0022", "1111", "1111"]
#     non_three_d_hand = ["102", "2", "122", "0022", "11", "1112", "1022", "2022", "0", "0122", "1", "1111"]
#     bad_char_hand = ["112z", "102t", "1f32", "0a32", "111c", "1112", "a022", "102b", "103a", "0032", "aaaa", "1111"]
#     four_pair_hand = ["0000", "0001", "0002", "0011", "1000", "1001", "1002", "1010", "1011", "1012", "1111", "2222"]
    
#     assert specs.count_sets(four_pair_hand) == 4

#     with pytest.raises(ValueError) as excinfo:
#         specs.count_sets(small_hand)
#     assert excinfo.value.args[0] == "there are not exactly 12 cards"

#     with pytest.raises(ValueError) as excinfo:
#         specs.count_sets(duplicate_hand)
#     assert excinfo.value.args[0] == "the cards are not all unique"

#     with pytest.raises(ValueError) as excinfo:
#         specs.count_sets(non_three_d_hand)
#     assert excinfo.value.args[0] == "one or more cards does not have exactly 4 digits"

#     with pytest.raises(ValueError) as excinfo:
#         specs.count_sets(bad_char_hand)
#     assert excinfo.value.args[0] == "one or more cards has a character other than 0, 1, or 2"


# def test_is_set():
#     a = "0120"
#     b = "1201"
#     c = "2012"
#     d = "1111" 
#     e = "1000"
#     f = "2000"
#     g = "0000"

#     assert specs.is_set(a, b, c) == True
#     assert specs.is_set(a, b, d) == False
#     assert specs.is_set(d, b, c) == False
#     assert specs.is_set(e, f, g) == True