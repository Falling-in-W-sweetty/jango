# _*_ coding: utf-8 _*_
# 2个函数，1个用了self，1个没有用self
class Test_A():
    def test_01(self):
        self.one = "hello, A!"
        print(self.one)

    def test_02(two):
        # two = "hello, B!"
        print(two)

a = Test_A()
a.test_01()
a.test_02()
print([i for i in range(10)])
# test_03()