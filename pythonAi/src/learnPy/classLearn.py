#学习类 
'''
python 也是面向对象的语言
'''
#创建一个momeryClass类， 里面有个属性x = 1
class momeryClass:
    x = 1

#创建一个momeryClass对象p1  
p1 = momeryClass()
print(p1.x)
        
'''
每个类都有一个__init__ 内建函数 ，所有类都有一个名为 __init__() 的函数，它始终在启动类时执行。
使用 __init__() 函数将值赋给对象属性，或者在创建对象时需要执行的其他操作

在类内部的函数， 第一个参数代表对类当前实例的引用，在方法定义的时候，不必一定命名为self， 外部调用不需要传
'''
class personClass:
    def __init__(self,name,age):
        self.name = name
        self.age = age

    def myFuction(self):
        print("have a nice day")
    def changeAge(abc,age):
        abc.age = age
        print('my age is:',abc.age)

#创建一个personClass对象 person1
person1 = personClass('ylx',32)
print(f"姓名:{person1.name}, 年龄:{person1.age}")

person1.myFuction()
person1.changeAge(33)

'''
继承
自动继承父类方法  ,添加父类同名方法 自动覆盖父类方法
super()  父类方法
'''
class student(personClass):
    def __init__(self, name, age,year):
        super().__init__(name, age)
        self.granduator = year

    def myFuction(self): #父类同名方法， 修改实现
        print('welcome to python, study hard pls')
    
    def learnPython(self):
        print('hello  world')
    

student1 = student('zhangsan',15,2014)
student1.myFuction()
student1.changeAge(19)
student1.learnPython()

'''

'''