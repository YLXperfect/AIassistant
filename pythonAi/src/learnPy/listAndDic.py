#列表学习  如果变量名跟内置方法名重复了， 再调用内置方法的时候会出错    list = [1,2,3]   list(list) 报错 规范变量名


list1 = ['111','222',333]
print(list1)

#可以通过索引访问 ，0 1，2   如果索引为负，代表倒序  -1 最后一个元素， -2 倒数第二个元素...
#索引超过list长度会报错

print(list1[0])
print(list1[-1])

#范围索引  [x:y]  表示从第x个元素到第Y个元素， 不包括第Y个元素  如果Y大于list的长度， 不会报错，会一直取到最后一个元素
print('第0-2项 不包括list1[2]',list1[0:2]) #第 【0-2)  项元素
print("从第0项开始所有元素",list1[0:])  #['111','222',333]

print("倒数两个元素-2:",list1[-2:]) #222


#判断list中是否包含某个元素 in  
x='222'
if x in list1:
    print('yes' ,x)

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

#Dic  字典 ， 跟别的语言的字典类似  key value
#字典是一个无序、可变和有索引的集合。在 Python 中，字典用花括号编写，拥有键和值。


thisdict =	{
  "brand": "Porsche",
  "model": "911",
  "year": 1963
}
print(thisdict)

#通过key访问对应的value   
print("model对应的值是:",thisdict['model'])
print(f"model对应的值是:{(thisdict['model'])}")
print(f"model对应的值是:{thisdict['model']}")

#也可以通过get( ) 获取
print("model对应的值是:",thisdict.get('model'))
#修改  直接修改key对应的值
thisdict['model'] = '718'
print("model对应的值是:",thisdict.get('model'))

#遍历  for x in dic 
#获取所有的key
for x in thisdict:
    print("key==",x)
#获取所有的值
for x in thisdict:
    print("value==",thisdict[x])

#也可以只用.values获取
for x in thisdict.values():
  print(x)


#。items 遍历键值
for x, y in thisdict.items():
  print(x, y)

#检查某个key是否存在
if 'model' in thisdict:
    print('yes model yes ')
else:
    print('No model  NO ')

#长度  len()
print("thisdic的长度",len(thisdict))

# 增删改
# 直接添加
thisdict['newKey'] = 'newValue'
# 直接修改key对应的值
thisdict['brand'] = 'Benz'
print(thisdict)

#删除key- value   pop() 删除对应的key
#popitem() 删除最后一项
#del  删除对应key 或者删除整个字典对象  del thisdict['brand']    /  del thisdict
'''
{'brand': 'Benz', 'model': '718', 'year': 1963, 'newKey': 'newValue'}
'''
thisdict.pop('newKey')
print(thisdict)
thisdict.popitem()
print(thisdict)    #{'brand': 'Benz', 'model': '718'}

del thisdict['brand']
print(thisdict) 
#clear  清空  
#字典复制  跟list一样  直接 dict2=dict1 复制是浅拷贝，  修改一个 ，两个都会变 用copy获取副本， 或者用内建方法dict()