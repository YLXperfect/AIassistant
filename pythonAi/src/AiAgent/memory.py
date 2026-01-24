#记忆类
'''
创建一个对话记忆类   封装获取  清空 添加 存储对话的方法
'''
class ConversationMemory:
    def __init__(self):
        self.memoryList = [] #初始化一个空对话列表

#添加消息
    def add_to_memory(self,role,content):
        message = {'role':role, 'content':content}
        self.memoryList.append(message)

#清空消息列表
    def clearList(self):
        self.memoryList.clear()

#获取消息列表
    def getAllMemoryList(self):
        return self.memoryList.copy()

#获取最近N条消息
    def getLastMemoryList(self,n):
        return self.memoryList[-n:].copy()

#获取消息总条数
    def getMessageCount(self):
        return len(self.memoryList)

#列表推导式  获取最近N条消息的内容
    def get_Recent_messages(self,n):
        recent_messageList = self.getLastMemoryList(n)
        #用推导式获取列表中消息内容   新列表 = [对元素的操作 for 元素 in 可迭代对象 if 条件]  if可选
        # contents = [msg['content'] for msg in recent_messageList]

        # return contents
        return recent_messageList
#获取最近N条用户消息
    def get_Recent_user(self,n):
        recent_messageList = self.getLastMemoryList(n)
        #用推导式获取列表中消息内容   新列表 = [对元素的操作 for 元素 in 可迭代对象 if 条件]  if可选
        role_content = [msg['content'] for msg in recent_messageList if msg['role']=='user']

        return role_content