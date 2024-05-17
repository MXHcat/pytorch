import numpy as np
from sklearn.model_selection import train_test_split
import jieba
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn import metrics
from sklearn.metrics import accuracy_score, recall_score, f1_score

#1.查看数据
h = open('ham_data.txt', encoding='utf-8')  # 正常邮件
s = open('spam_data.txt', encoding='utf-8') # 垃圾邮件

h_data = h.readlines()
s_data = s.readlines()
print(h_data[0:1]) # 查看一下前 1 封邮件为例
print(s_data[0:1])

#2.数据预处理
#2.1 生成样本和标签
h_labels = np.ones(len(h_data)).tolist() # 生成全 1 的正标签list
s_labels = np.zeros(len(s_data)).tolist() # 生成全 0 的负标签list
# 拼接正负样本集和标签集合到一起
datas = h_data + s_data
labels = h_labels + s_labels
#2.2 划分训练集和测试集
#datas：样本集,labels：标签集,train_test_split：划分到测试集的比例,random_state：随机种子，取同一个的随机种子那么每次划分出的测试集是一样的。
#返回值的含义：train_d：训练集,test_d：测试集,train_y：训练标签,test_y：测试标签
train_d, test_d, train_y, test_y = train_test_split(datas, labels, test_size=0.25, random_state=5)
print(train_y[0:10])
#2.3 分词
def tokenize_words(corpus):
    tokenized_words = jieba.cut(corpus) # 调用 jieba 分词
    tokenized_words = [token.strip() for token in tokenized_words] # 去掉回车符，转为list类型
    return tokenized_words
#2.4 去除停用词
def remove_stopwords(corpus): # 函数输入为全部样本集（包括训练和测试）
    sw = open('stop_word.txt', encoding='utf-8') # 加载停用词表
    sw_list = [l.strip for l in sw] # 去掉回车符存放至list中
    # 调用分词函数
    tokenized_data = tokenize_words(corpus)
    # 使用list生成式对停用词进行过滤
    filtered_data = [data for data in tokenized_data if data not in sw_list]
    # 用' '将 filtered_data 串起来赋值给 filtered_datas（不太好介绍，可以看一下下面处理前后的截图对比）
    filtered_datas = ' '.join(filtered_data)
    # 返回是去除停用词后的字符串
    return filtered_datas

def preprocessing_datas(datas):
    preprocessing_datas = []
    # 对 datas 当中的每一个 data 进行去停用词操作
    # 并添加到上面刚刚建立的 preprocessed_datas 当中
    for data in tqdm(datas):
        data = remove_stopwords(data)
        preprocessing_datas.append(data)
    # 返回预处理后的样本集
    return preprocessing_datas

pred_train_d = preprocessing_datas(train_d)
print(pred_train_d[0])
pred_test_d = preprocessing_datas(test_d)
print(pred_test_d[0])

#3. 特征提取
# min_df： 忽略掉词频严格低于定阈值的词。
# norm ：标准化词条向量所用的规范。
# smooth_idf：添加一个平滑 IDF 权重，即 IDF 的分母是否使用平滑，防止 0 权重的出现。
# use_idf： 启用 IDF 逆文档频率重新加权。
# ngram_range：同词袋模型
vectorizer = TfidfVectorizer(min_df=1, norm='l2', smooth_idf=True, use_idf=True, ngram_range=(1, 1))
tfidf_train_features = vectorizer.fit_transform(pred_train_d)
# print(tfidf_train_features)
# print(np.array(tfidf_train_features.toarray()).shape)
tfidf_test_features = vectorizer.transform(pred_test_d)
print(tfidf_test_features)

#4. 分类
svm = SGDClassifier(loss='hinge')
svm.fit(tfidf_train_features, train_y)
SGDClassifier()
#查看分类结果
predictions = svm.predict(tfidf_test_features)
print(predictions)

#5. 评估算法
# 计算准确率
accuracy_score = np.round(metrics.accuracy_score(test_y, predictions), 2)
print("accuracy:",accuracy_score)
# 计算召回率
recall = np.round(recall_score(test_y, predictions), 2)
print("Recall:", recall)
# 计算F1值
f1 = np.round(f1_score(test_y, predictions), 2)
print("F1 Score:", f1)

#6. 验证
try:
  id = int(input('请输入样本编号：'))
  if 0 <= id < len(test_y):
    print('邮件类型：', '垃圾邮件' if test_y[id] == 0 else '正常邮件')
    print('预测邮件类型：', '垃圾邮件' if predictions[id] == 0 else '正常邮件')
    print('文本：', test_d[id].strip())  # 去除文本两侧的空白字符
  else:
    print('输入的编号无效！')
except ValueError:
  print('请输入一个有效的整数编号！')





