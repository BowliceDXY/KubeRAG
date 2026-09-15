"""计算器工具：AST 白名单安全求值冒烟测试"""
from app.agent import calculator


def test_basic_arithmetic():
    assert calculator("2 + 3 * 4") == "计算结果：2 + 3 * 4 = 14"


def test_parentheses():
    assert calculator("(10 + 5) / 3") == "计算结果：(10 + 5) / 3 = 5.0"


def test_unary_minus():
    assert calculator("-5 + 3") == "计算结果：-5 + 3 = -2"


def test_divide_by_zero():
    assert "除数不能为 0" in calculator("1 / 0")


def test_syntax_error():
    assert "语法不正确" in calculator("2 +")


def test_reject_function_call():
    # 白名单外节点（函数调用/属性访问/下标）必须被拒绝，不能执行任意代码
    assert "非法字符" in calculator("__import__('os').system('dir')")
    assert "非法字符" in calculator("len([1, 2, 3])")
    assert "非法字符" in calculator("(1).__class__")
