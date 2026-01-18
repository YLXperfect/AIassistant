#列表学习  如果变量名跟内置方法名重复了， 再调用内置方法的时候会出错    list = [1,2,3]   list(list) 报错 规范变量名
list1 = ['111','222',333]
print(list1)

#可以通过索引访问 ，0 1，2   如果索引为负，代表倒序  -1 最后一个元素， -2 倒数第二个元素...
#索引超过list长度会报错

print(list1[0])
print(list1[-1])

#范围索引  [x:y]  表示从第x个元素到第Y个元素， 不包括第Y个元素  如果Y大于list的长度， 不会报错，会一直取到最后一个元素
print(list1[0:4])  #['111','222',333]

print(list1[-2:-1]) #222


#判断list中是否包含某个元素 in  
x='222'
if x in list1:
    print('yes' +x)

#遍历list  for  in   获取list长度用 len()
for i in list1:
    print(i)


#增删改查 

#增加  append  在末尾增加， insert  在对应索引插入
list1.append('python')
list1.insert(1,"Tesla")
print(list1)
#更改  直接使用索引
list1[2]=666
print(list1[2])

#删除 remove + 元素   pop() 删除指定索引， 无参数默认删除最后一项
#del + list[i] 删除对应索引元素 或者 del + list  删除全部包括,list变量
#clear  清空列表  list变量还在

list1.remove('111')
print(list1)

list1.pop(1)
print(list1)

list1.pop()
print(list1)

# del list1
# print(list1) #报错 ，list1不存在了

list1.clear()
print(list1)

#复制列表  list2 = list1 ,浅拷贝， 修改其中一个，另一个也会改变
list222 = [1,2,3,4]
list2= list222
list2[0] = 0
print(list222,list2)

#副本，修改list  testList不会被修改   
testList = list222.copy()
list222[0] = 1
print(list222,testList)

list222 = [1,2,3,4]
testList2 = list(list222)
list222[3] = 1
print(list222,testList2)

# thislist = ["apple", "banana", "cherry"]
# mylist = list(thislist)
# print(mylist)

