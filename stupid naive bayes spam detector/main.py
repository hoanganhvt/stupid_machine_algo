import pandas as pd
import numpy as np
import re


def message_clean(msg):
    tmp_msg = msg.lower()
    tmp_msg = re.sub(r'[^a-z ]', '', tmp_msg)
    return ' '.join(tmp_msg.split())  

def convert_to_bag_of_words(msg,word_list:set):
    res={}
    for w in word_list:
        res[w]=0
    for w in msg.split(' '):
        res[w]=1
    return res

def get_parameter(word_bag_list,word_list):
    word_list_a=list(word_list)
    phi_y=0.0
    phi_x_y1=[0 for i in range(len(word_list))]
    phi_x_y0=[0 for i in range(len(word_list))]

    #calculate phi_x_y1
    for j in range(len(phi_x_y1)):
        first=0
        second=0
        for i in range(len(word_bag_list)):
            first=first+float(word_bag_list[i]["label"]==1 and word_bag_list[i]["bag"][word_list_a[j]]==1)
            second=second+float(word_bag_list[i]["label"]==1)
        phi_x_y1[j]=first/second

    #calculate phi_x_y0
    for j in range(len(phi_x_y1)):
        first=0
        second=0
        for i in range(len(word_bag_list)):
            first=first+float(word_bag_list[i]["label"]==0 and word_bag_list[i]["bag"][word_list_a[j]]==1)
            second=second+float(word_bag_list[i]["label"]==0)
        phi_x_y0[j]=first/second
    
    #calculate phi_y
    for i in range(len(word_bag_list)):
        phi_y+=float(word_bag_list[i]["label"]==1)
    phi_y = phi_y/len(word_bag_list)
    
    return [phi_x_y1,phi_x_y0,phi_y]
        

def hypothesis(word_bag,phi_x_y1,phi_x_y0,phi_y):
    word_list_a=list(word_list)
    res=1
    if word_bag["label"]==1:
        res=res*phi_y
    else:
        res=res*(1-phi_y)
        
    mini_res=1
    for j in range(len(word_bag["bag"])):
        if word_bag["label"]==1:
            if word_bag["bag"][word_list_a[j]]==1:
                mini_res=mini_res*phi_x_y1[j]
            else:
                mini_res=mini_res*(1-phi_x_y1[j])
        else:
            if word_bag["bag"][word_list_a[j]]==1:
                mini_res=mini_res*phi_x_y0[j]
            else:
                mini_res=mini_res*(1-phi_x_y0[j])
    res*=mini_res

    return res 

data=pd.read_csv('data.csv')

word_list=set()
msg_list=[{"message":message_clean(data['Message'][i]),"label":1 if data['Category'][i]=="spam" else 0} for i in range(len(data))]

for i in range(len(msg_list)):
    for w in msg_list[i]["message"].split(' '):
        word_list.add(w)

msg_list=msg_list[:len(msg_list)//2]
word_bag_list=[{"bag":convert_to_bag_of_words(msg_list[i]["message"],word_list),"label":msg_list[i]["label"]} for i in range(len(msg_list))]

phi_x_y1,phi_x_y0,phi_y=get_parameter(word_bag_list,word_list)

#this part is written by claude cuz as usual, im too lazy : )
# ===== PHẦN TEST VỚI DATA KHÔNG ĐƯỢC TRAIN =====
print("\n===== TESTING VỚI DATA KHÔNG ĐƯỢC TRAIN =====")

# Lấy nửa sau của data (phần không được train)
test_msg_list = [{"message":message_clean(data['Message'][i]),"label":1 if data['Category'][i]=="spam" else 0} for i in range(len(data)//2, len(data))]

# Chuyển test data thành bag of words
test_word_bag_list = [{"bag":convert_to_bag_of_words(test_msg_list[i]["message"],word_list),"label":test_msg_list[i]["label"]} for i in range(len(test_msg_list))]

# Predict và tính accuracy
correct = 0
total = len(test_word_bag_list)

for i in range(len(test_word_bag_list)):
    # Tính xác suất cho spam (label=1) và ham (label=0)
    test_bag_spam = {"bag": test_word_bag_list[i]["bag"], "label": 1}
    test_bag_ham = {"bag": test_word_bag_list[i]["bag"], "label": 0}
    
    prob_spam = hypothesis(test_bag_spam, phi_x_y1, phi_x_y0, phi_y)
    prob_ham = hypothesis(test_bag_ham, phi_x_y1, phi_x_y0, phi_y)
    
    # Predict: chọn label có xác suất cao hơn
    predicted_label = 1 if prob_spam > prob_ham else 0
    actual_label = test_word_bag_list[i]["label"]
    
    if predicted_label == actual_label:
        correct += 1

# In kết quả
accuracy = (correct / total) * 100
print(f"Số lượng test samples: {total}")
print(f"Số lượng dự đoán đúng: {correct}")
print(f"Accuracy: {accuracy:.2f}%")

# Test một vài ví dụ cụ thể
print("\n===== MỘT VÀI VÍ DỤ CỤ THỂ =====")
for i in range(min(5, len(test_word_bag_list))):
    test_bag_spam = {"bag": test_word_bag_list[i]["bag"], "label": 1}
    test_bag_ham = {"bag": test_word_bag_list[i]["bag"], "label": 0}
    
    prob_spam = hypothesis(test_bag_spam, phi_x_y1, phi_x_y0, phi_y)
    prob_ham = hypothesis(test_bag_ham, phi_x_y1, phi_x_y0, phi_y)
    
    predicted = "SPAM" if prob_spam > prob_ham else "HAM"
    actual = "SPAM" if test_word_bag_list[i]["label"] == 1 else "HAM"
    
    print(f"\nMessage {i+1}: {test_msg_list[i]['message'][:50]}...")
    print(f"Predicted: {predicted}, Actual: {actual}, {'✓' if predicted == actual else '✗'}")