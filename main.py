# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import io
import streamlit as st
from PIL import Image
import numpy as np
import torch
from torch import device
import streamlit as st
import pickle
import pandas as pd
st.title('Заполните данные о своем здоровье')

answer_diabets = None
answer_BP = None
answer_HighChol = None
answer_CholCheck = None
answer_BMI = None
answer_Smoker = None
answer_Stroke = None
answer_HeartDiseaseorAttack = None
answer_PhysActivity = None
answer_Fruits = None
answer_Veggies = None
answer_HvyAlcoholConsump = None
answer_AnyHealthcare = None
answer_NoDocbcCost = None
answer_GenHlth = None
answer_MentHlth = None
answer_PhysHlth = None
answer_DiffWalk = None
answer_Sex = None
answer_Age = None
answer_Education = None
answer_Income = None
selected_option = None

# Заголовок страницы

# Вопрос о поле
Sex = st.selectbox("Ваш пол?", ('Женский', 'Мужской'))
if Sex == 'Мужской':
    answer_Sex = 1
elif Sex == 'Женский':
    answer_Sex = 0

# Вопрос о возрасте
Age = st.selectbox("К какой возрастной группе вы относитесь?", ('18-24', '25-29', '30-34', '35-39', '40-44', '45-49', '50-54', '55-59', '60-64', '65-69', '70-74', '75-79', '80 и старше'))
age_mapping = {
    '18-24': 1, '25-29': 2, '30-34': 3, '35-39': 4, '40-44': 5, '45-49': 6,
    '50-54': 7, '55-59': 8, '60-64': 9, '65-69': 10, '70-74': 11, '75-79': 12, '80 и старше': 13
}
answer_Age = age_mapping.get(Age)


# Вопрос о наличии диабета
diabetes_status = st.selectbox("Есть ли у вас подтвержденный диабет?", ( 'Нет диабета', 'Преддиабетное состояние', 'Диабет подтвержденный'))
if diabetes_status == 'Нет диабета':
    answer_diabets = 0
elif diabetes_status == 'Преддиабетное состояние':
    answer_diabets = 1
elif diabetes_status == 'Диабет подтвержденный':
    answer_diabets = 2

# Вопрос о наличии высокого давления

HighBP = st.selectbox("Бывает ли у вас высокое давление?", ('Нет', 'Да'))
if HighBP == 'Да':
    answer_BP = 1
elif HighBP == 'Нет':
    answer_BP = 0

# Вопрос о наличии высокого уровня холестирина
HighChol = st.selectbox("Высокий ли у вас уровень холестирина?", ('Нет', 'Да'))
if HighChol == 'Да':
    answer_HighChol = 1
elif HighChol == 'Нет':
    answer_HighChol = 0

# Вопрос о сдаче анализов на холестирин
CholCheck = st.selectbox("Когда в последний раз вы сдавали анализ на холестирин?", ('Сдавал за последние 5 лет', 'Сдавал более 5 лет назад'))
if CholCheck == 'Сдавал за последние 5 лет':
    answer_CholCheck = 1
elif CholCheck == 'Сдавал более 5 лет назад':
    answer_CholCheck = 0

# Вопрос про индекс массы тела
BMI = st.text_input('Введите индекс массы тела (BMI):')
if BMI and not BMI.replace('.', '').isdigit():
    st.warning('Пожалуйста, введите корректное число для индекса массы тела.')
else: answer_BMI = BMI

#Вопрос про курение
Smoker= st.selectbox("Выкурили ли вы за всю жизнь более 100 сигарет(5 пачек)?", ('Нет', 'Да'))
if Smoker == 'Да':
    answer_Smoker = 1
elif Smoker == 'Нет':
    answer_Smoker = 0

#Вопрос про инсульт
Stroke = st.selectbox("Был ли у вас инсульт?", ('Нет', 'Да'))
if Stroke == 'Да':
    answer_Stroke = 1
elif Smoker == 'Нет':
    answer_Stroke = 0

#Вопрос про ишемическую болезнь сердца и имфаркт
HeartDiseaseorAttack = st.selectbox("Был ли у вас инфаркт или ишемическая болезнь сердца?", ('Нет', 'Да'))
if HeartDiseaseorAttack == 'Да':
    answer_HeartDiseaseorAttack = 1
elif HeartDiseaseorAttack == 'Нет':
    answer_HeartDiseaseorAttack = 0

#Вопрос про физическую активность
PhysActivity = st.selectbox("Была ли у вас физическая активность кроме работы за последние 30 дней?", ('Нет', 'Да'))
if PhysActivity == 'Да':
    answer_PhysActivity = 1
elif PhysActivity == 'Нет':
    answer_PhysActivity = 0

# Вопрос про потребление фруктов
Fruits = st.selectbox("Потребляете ли фрукты 1 раз в день или более?", ('Нет', 'Да'))
if Fruits == 'Да':
    answer_Fruits = 1
elif Fruits == 'Нет':
    answer_Fruits = 0

# Вопрос про потребление овощей
Veggies = st.selectbox("Потребляете ли овощи 1 раз в день или более?", ('Нет', 'Да'))
if Veggies == 'Да':
    answer_Veggies = 1
elif Veggies == 'Нет':
    answer_Veggies = 0

# Вопрос про употребление алкоголя в больших количествах
HvyAlcoholConsump = st.selectbox("Употребляете ли вы большое количество алкоголя?", ('Нет', 'Да'))
if HvyAlcoholConsump == 'Да':
    answer_HvyAlcoholConsump = 1
elif HvyAlcoholConsump == 'Нет':
    answer_HvyAlcoholConsump = 0

# Вопрос о наличии медицинской страховки
AnyHealthcare = st.selectbox("Имеете ли вы какую-либо медицинскую страховку кроме ОМС?", ('Нет', 'Да'))
if AnyHealthcare == 'Да':
    answer_AnyHealthcare = 1
elif AnyHealthcare == 'Нет':
    answer_AnyHealthcare = 0

# Вопрос о том, были ли случаи, когда из-за стоимости вы не могли позволить себе посетить врача
NoDocbcCost = st.selectbox("Были ли случаи за последние 12 месяцев, когда вам нужно было обратиться к врачу, но вы этого не сделали?", ('Нет', 'Да'))
if NoDocbcCost == 'Да':
    answer_NoDocbcCost = 1
elif NoDocbcCost == 'Нет':
    answer_NoDocbcCost = 0

# Вопрос о общем состоянии здоровья
GenHlth = st.slider("Как бы вы оценили ваше общее здоровье по шкале от 1 до 5?", 1, 5)
answer_GenHlth = GenHlth

# Вопрос о психическом здоровье
MentHlth = st.slider("Сколько дней за последние 30 дней ваше психическое здоровье было плохим?", 1, 30)
answer_MentHlth = MentHlth

# Вопрос о физическом здоровье
PhysHlth = st.slider("Сколько дней за последние 30 дней ваше физическое здоровье было плохим?", 1, 30)
answer_PhysHlth = PhysHlth

# Вопрос о трудностях при ходьбе
DiffWalk = st.selectbox("Имеете ли вы серьезные трудности с ходьбой или подъемом по лестнице?", ('Нет', 'Да'))
if DiffWalk == 'Да':
    answer_DiffWalk = 1
elif DiffWalk == 'Нет':
    answer_DiffWalk = 0


# Вопрос об образовании
Education = st.selectbox("Ваш уровень образования?", ('Никогда не учился или только детский сад', 'Неоконченное основное общее обарзование(1-8 класс)', 'Неоконченное среднее общее образование(9-10 класс)', 'Среднее общее образование(окончание 11 классов)', 'Незаконченное высшее образование', 'Высшее образование(бакалавриат, специалитет)'))
education_mapping = {
    'Никогда не учился или только детский сад': 1, 'Неоконченное основное общее обарзование(1-8 класс)': 2,
    'Неоконченное среднее общее образование(9-10 класс)': 3, 'Среднее общее образование(окончание 11 классов)': 4,
    'Незаконченное высшее образование': 5, 'Высшее образование(бакалавриат, специалитет)': 6
}
answer_Education = education_mapping.get(Education)

# Вопрос о доходе
Income = st.selectbox("Ваш ежемесячный доход?", ('Менее 15000 рублей', '15,000 - 22000 рублей', '22001 - 30000 рублей', '30001 - 37500 рублей', '37501 - 52000 рублей', '52001 - 75000 рублей', '75001 - 112000 рублей', 'более 112000 рублей'))
income_mapping = {
    'Менее 15000 рублей': 1, '15,000 - 22000 рублей': 2, '22001 - 30000 рублей': 3,
    '30001 - 37500 рублей': 4, '37501 - 52000 рублей': 5, '52001 - 75000 рублей': 6,
    '75001 - 112000 рублей': 7, 'более 112000 рублей': 8
}
answer_Income = income_mapping.get(Income)
#loaded_model = Net()
#loaded_model.load_state_dict(torch.load('diabetes_model.pth'))
#loaded_model.eval()
result = st.button('Отправить данные')
def predict(model, input_data):
    model.eval()
    with torch.no_grad():
        # Предобработка входных данных (предполагается, что input_data - это список)
        input_tensor = torch.FloatTensor(input_data).unsqueeze(0).to(device)  # Преобразование в тензор и добавление размерности батча

        # Получение предсказания от модели
        output = model(input_tensor)

        # Извлечение индекса класса с наивысшей вероятностью
        prediction = torch.argmax(output, dim=1).item()

    return prediction
with open('model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)
input_data = [1.0, 0.0, 0.0, 1.0, 26.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 20.0, 3.0]
prediction = predict(model, input_data)

st.title(prediction)
print(f'Model Prediction: {prediction}')
#st.title(answer_diabets)
#st.title(answer_BP)
#st.title(answer_HighChol)
#st.title(answer_CholCheck)
#st.title(answer_BMI)
#st.title(answer_Smoker)
#st.title(answer_Stroke)
#st.title(answer_HeartDiseaseorAttack)
#st.title(answer_PhysActivity)




#def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
  #  print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
#if __name__ == '__main__':
    #print_hi('PyCharm')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
