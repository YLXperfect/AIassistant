#函数学习 没有参数

def my_print():
    print("2222")

my_print()


#有默认参数的函数 ，不传参的时候 默认是china
def canshuHanshu(country='china'):
    print("I am from" +country)

canshuHanshu()
canshuHanshu('America')


x = 3

#作用域 局部变量 全局变量
def add(x):
    x = x+3
    y = 4
    return x+y

print(add(x))
print(x)


#多个参数 
def my_sum(x,y):
    return x+y

print(my_sum(3,4))

#不定长参数  *args 可以传入任意个参数
def my_sum(*args):
    return sum(args)

print(my_sum(1,2,3,4,5))

#关键字参数    在调用函数的时候 直接给对应参数赋值， 顺序无关
def test1111(canshu1,canshu2,canshu3):
    print(canshu1, canshu2, canshu3)
test1111(canshu1="2", canshu3="3",canshu2="5")


# 不可变变量 数值  字符串等， 可变变量 列表 字典 集合
#不可变变量在函数内部会复制一份操作， 跟全局变量内存地址不一样， 可变变量会也会复制一份，但是内存地址相同，指向同一变量
tempList = ['1','2','3']
x = 10
print(f"函数外x的id:{id(x)}")
print(f"函数外list的id:{id(tempList)}")
def my_clear(argList,x):
    argList.clear()
    x += 10
    print(f"函数内x的id:{id(x)}")
    print(f"函数内list的id:{id(tempList)}")

my_clear(tempList,x)
print(tempList)
print(x)
