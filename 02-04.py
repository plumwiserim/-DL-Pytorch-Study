# 02. 파이토치 기초(Pytorch Basic)
# 02-04. 파이썬 클래스(class)

# 함수와 클래스 차이
# 1. 함수 
result = 0
def add(num): 
    global result
    result += num 
    return result 

print(add(3))   # 3
print(add(4))   # 7

result1 = 0
result2 = 0

def add1(num):
    global result1
    result1 += num
    return result1

def add2(num):
    global result2
    result2 += num
    return result2

print(add1(3))  # 3
print(add1(4))  # 7
print(add2(3))  # 3
print(add2(7))  # 10

# 2. 클래스 
class Calculator:
    def __init__(self): # 객체 생성 시 호출될 때 실행되는 초기화 함수. 이를 생성자라고 한다. 
        self.result = 0
    
    def add(self, num): # 객체 생성 후 사용할 수 있는 함수
        self.result += num
        return self.result
    
cal1 = Calculator()
cal2 = Calculator()

print(cal1.add(3))  # 3
print(cal1.add(4))  # 7
print(cal2.add(3))  # 3
print(cal2.add(7))  # 10

# 두 개의 객체가 독립적으로 연산되고 있음
# 함수로 구현하려고 했다면, 같은 기능의 함수를 두 개 만들어야 하지만
# 클래스로 구현한다면, 하나의 클래스를 선언하고, 이 클래스를 통해 별도의 객체들을 생성하면 되기 때문에 코드가 훨씬 간결해진다. 
