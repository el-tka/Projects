#!/usr/bin/env python
# coding: utf-8

# <div style="border:solid orange 2px; padding: 20px"> <h7 style="color:black; margin-bottom:20px">Привет! Меня зовут Миша, я буду проводить ревью на этом спринте. 
# Мои комментарии и замечания далее по тексту помечены <font color='orange'>цветом и рамкой</font>. Постарайся учесть их в ходе выполнения дальнейших проектов. Удачи!</h7>

# ## Исследование рынка общественного питания Москвы

# 1. [Открытие данных](#start)
# 2. [Предобработка данных](#preprocessing)
#     * [Обработка дубликатов](#duplicates)
#     * [Обработка пропущенных значений](#null)
# 3. [Анализ рынка общественного питания](#analysis)
#     * [Исследование соотношения видов объектов общественного питания по количеству](#objectsamount)
#     * [Исследование соотношения сетевых и несетевых заведений по количеству](#chain)
#     * [Исследование соотношения вида объекта общественного питания и его вида распределения](#objecttype)
#     * [Исследование распределения количества посадочных мест для сетевых заведений](#number)
#     * [Исследование распределения количества посадочных мест по видам объектов питания](#numbertype)
#     * [Обработка адресов заведений общественного питания](#address)
#     * [Анализ наиболее популярных районов для заведений](#populardistrict)
#     * [Анализ наименее популярных районов для заведений](#unpopulardistrict)
#     * [Анализ распределения количества посадочных мест для наиболее популярных районов](#numberforpopular)
# 4. [Общие выводы](#insights)
# 5. [Рекомендации](#recommendations)
# 6. [Презентация](#presentation)

# #### Part 1

# <a id="start"></a>

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns


# In[2]:


data = pd.read_csv('/datasets/rest_data.csv')


# #### Part 2

# <a id="preprocessing"></a>

# In[3]:


data.info()


# In[4]:


data['object_name'] = data['object_name'].str.lower()


# In[5]:


data.head()


# <a id="duplicates"></a>

# In[6]:


data.duplicated().sum()


# <a id="null"></a>
# 

# In[7]:


data.isnull().sum()


# #### Part 3

# <a id="objectsamount"></a>

# In[8]:


#Исследуем соотношение видов объектов общественного питания по количеству
data['object_type'].value_counts()


# In[9]:


plt.style.use('seaborn')


# In[10]:


fig, ax = plt.subplots()
data['object_type'].value_counts().plot(kind='bar', figsize = (9,6))
ax.set_ylabel('Количество')
ax.legend()
ax.set_title('Объекты общественного питания')
plt.xticks(rotation=90)


# На графике видно, что наиболее распространенными объектами питания являются кафе, столовые и рестораны в Москве

# <div style="border:solid orange 2px; padding: 20px"> <h7 style="color:black; margin-bottom:20px"><font color='orange'> Здесь всё хорошо

# <a id="chain"></a>

# In[11]:


#Исследуем соотношение сетевых и несетевых заведений по количеству
data['chain'].value_counts()


# In[12]:


from plotly import graph_objects as go
titles = ['Несетевой объект', 'Сетевой объект']
values = data['chain'].value_counts()
fig = go.Figure(data=[go.Pie(labels=titles, values=values)])
fig.update_traces(textposition='inside', textinfo='percent+label')
fig.update_layout(title_text='Cоотношение сетевых и несетевых объектов')
fig.show()


# На графике видно, что несетевые объекты питания составляют 80,7%, сетевые - 19,3%

# <div style="border:solid orange 2px; padding: 20px"> <h7 style="color:black; margin-bottom:20px"><font color='orange'>Верный результат

# <a id="objecttype"></a>

# In[13]:


#рассмотрим долю сетевых по видам объектов
data_by_object_chain = pd.pivot_table(data, index = 'object_type', columns = 'chain', values = 'id', aggfunc = 'count')


# In[14]:


data_by_object_chain['total'] = data_by_object_chain['да']+data_by_object_chain['нет']
data_by_object_chain['chain_share'] = (data_by_object_chain['да']/data_by_object_chain['total']*100).round(2)
data_by_object_chain['no_chain_share'] = (data_by_object_chain['нет']/data_by_object_chain['total']*100).round(2)
data_by_object_chain = data_by_object_chain.sort_values(by = 'chain_share', ascending = False)


# In[15]:


data_by_object_chain = data_by_object_chain.reset_index()


# In[16]:


data_by_object_chain


# In[17]:


import plotly.graph_objects as go
object_type = data_by_object_chain['object_type']

fig = go.Figure(data=[
    go.Bar(name='Сетевые', x=object_type, y= data_by_object_chain['chain_share']),
    go.Bar(name='Несетевые', x=object_type, y= data_by_object_chain['no_chain_share'])
           ])
fig.update_layout(barmode='group')
fig.update_layout(title_text='Виды распределения по объектам общественного питания')
fig.show()


# На графике видно, что наибольшей долей сетевого распределения обладают предприятия быстрого обслуживания (41,13%), далее по величине доли сетевых объектов следуют магазины (отделы кулинарии), рестораны и кафе с долей в 28,57%, 23,81%, 22,89% соответственно.

# <div style="border:solid orange 2px; padding: 20px"> <h7 style="color:black; margin-bottom:20px"><font color='orange'> Всё правильно

# <a id="number"></a>

# In[18]:


#найдем общее количество сетевых заведений
data.groupby('chain')['id'].sum()


# In[19]:


#выделим сетевые заведения в отдельный датасет
data_chain = data.query('chain == "да"' )


# In[20]:


data_chain.head()


# In[21]:


data_chain_count = data_chain.groupby('object_name').agg({'id':'count', 'number':'mean'}).sort_values(by='id',ascending=False)


# In[22]:


data_chain_count.columns = ['object_count', 'seat_amount']


# In[23]:


data_chain_count


# In[24]:


f, ax = plt.subplots(figsize=(6.5, 6.5))
sns.despine(f, left=True, bottom=True)
sns.scatterplot(x="object_count", y="seat_amount",
                palette="ch:r=-.2,d=.3_r",
                sizes=(1, 8), linewidth=0,
                data=data_chain_count, ax=ax)
ax.set_title('Распределение посадочных мест по количеству заведений')


# In[25]:


sns.jointplot(x=data_chain_count['object_count'], y=data_chain_count['seat_amount'])


# На графике распределения видно, что для сетевых объектов с большим количеством заведений характерно небольшое число посадочных мест. Для сетевых объектов с небольшим количеством заведений число посадочных мест разнится.

# <div style="border:solid brown 2px; padding: 20px"> <h7 style="color:black; margin-bottom:20px"><font color='brown'> Вывод корректный, но он должен ссылаться на полученные в ходе анализа данные. "Небольшое" число посадочных мест - это сколько?

# <a id="numbertype"></a>

# In[26]:


#найдем среднее количество посадочных мест по объектам 
data.groupby('object_type')['number'].mean().round(0).sort_values(ascending=False)


# In[27]:


data.groupby('object_type')['number'].mean().round(0).sort_values(ascending=False).plot(kind='bar')
plt.xticks(rotation=90)
plt.title('Среднее количество посадочных мест по объектам')
plt.legend()


# In[28]:


#рассмотрим вариацию числа посадочных мест по объектам
f, ax = plt.subplots(figsize=(15, 7))
sns.boxplot(x="object_type", y="number", data=data.query('number <= 500'))
plt.xticks(rotation = 30)


# На графике распределения посадочных мест по объектам питания видно, что среди объектов питания наибольшим числом посадочных мест обладают рестораны и столовые. Средннее число посадочных мест в столовых равно 130 шт, в ресторанах 97 шт.

# <div style="border:solid orange 2px; padding: 20px"> <h7 style="color:black; margin-bottom:20px"><font color='orange'> Здесь всё хорошо

# <a id="address"></a>

# In[29]:


#найдем топ 10 улиц по количеству заведений
data.head()


# In[30]:


data['street'] = data.address.str.split(',').str[1].str.strip()


# In[31]:


data['city'] = data.address.str.split(',').str[0].str.strip()


# In[32]:


data = data.query('city == "город Москва"')


# <a id="populardistrict"></a>

# In[33]:


top10street = data.query('street != "город Зеленоград"').groupby('street')['id'].count().sort_values(ascending =False).head(10)


# In[34]:


top10street = top10street.reset_index()


# In[35]:


top10street


# In[36]:


top10street.plot(x = 'street', y = 'id', kind='bar')
plt.xticks(rotation=40)
plt.title('ТОП 10 улиц по количеству заведений')
plt.legend()


# <div style="border:solid orange 2px; padding: 20px"> <h7 style="color:black; margin-bottom:20px"><font color='orange'> Результат верный

# In[37]:


import requests
import json
from bs4 import BeautifulSoup


# In[38]:


#напишем функцию для парсинга координат из геокодера по топ 10 улиц
apikey= 'a2511768-076c-4b08-8701-3d8ab2671625'


# In[39]:


def get_coordinates(row):
    apikey = 'a2511768-076c-4b08-8701-3d8ab2671625'
    url = 'https://geocode-maps.yandex.ru/1.x'
    params = {'geocode': row, 'apikey': apikey, 'results':1}
    response = requests.get(url, params=params)
    soup=BeautifulSoup(response.text, 'lxml')
    coordinates = soup.find('pos')
    try:
        return coordinates.text
    except:
        return 'not found'


# In[40]:


top10street['coordinates'] = top10street['street'].apply(get_coordinates)


# In[41]:


top10street


# In[42]:


#напишем функцию для парсинга полного адреса из геокодера по координатам и вычленим из полученных данных районы
def get_district(street):
    district_list = []
    apikey = 'a2511768-076c-4b08-8701-3d8ab2671625'
    url = 'https://geocode-maps.yandex.ru/1.x'
    coordinates = get_coordinates(street)
    params = {'geocode': coordinates, 'apikey': apikey, 'format':'xml', 'kind':'district'}
    response = requests.get(url, params=params)
    soup=BeautifulSoup(response.text, 'lxml')
    district = soup.find_all('dependentlocalityname')
    for row in district:
        district_list.append(row.text)
    try:
        return district_list[1]
    except:
        return district_list


# In[43]:


top10street['district'] = top10street['coordinates'].apply(get_district)


# In[44]:


top10street


# In[45]:


top10street_show = top10street[['street', 'district', 'id']]
top10street_show.columns = ['Улица', 'Район', 'Количество заведений']


# In[46]:


top10street_show


# В таблице определены районы для топ 10 улиц с наибольшим количеством заведений

# <a id="unpopulardistrict"></a>

# In[47]:


#найдем улицы с одним заведением  и определим для них районы из геокодера
data1street = data.groupby('street')['id'].count().sort_values(ascending = True).reset_index()
data1street = data1street[data1street['id']== 1]
data1street


# In[48]:


data1street['coordinates'] = data1street['street'].apply(get_coordinates)


# In[49]:


data1street['district'] = data1street['coordinates'].apply(get_district)


# In[50]:


data1street


# In[51]:


data1street['district'] = data1street['district'].astype('str')


# In[52]:


data1street_show = data1street.query('district!="[]"').groupby('district')['id'].count().sort_values(ascending=False).head(10)


# In[53]:


data1street_show = data1street_show.reset_index()


# In[54]:


data1street_show.columns =['Район', "Количество улиц с одним заведением"]


# In[55]:


data1street_show


# В таблице определены районы с наибольшим количеством улиц, где находится лишь одно заведение.

# <a id="numberforpopular"></a>

# In[56]:


#отсортируем данные, оставив лишь улицы с большим количеством заведений
top10street


# In[57]:


data_top_street = data[data['street'].isin(top10street['street'])]


# In[58]:


data_top_street


# In[59]:


f, ax = plt.subplots(figsize=(9, 6.5))
sns.distplot(data_top_street['number'], ax=ax)
ax.set_title('Распределение количества мест в объектах для улиц с большим числом заведений')


# In[60]:


data_top_street_clear = data_top_street.query('number<300')


# In[61]:


f, ax = plt.subplots(figsize=(9, 6.5))
sns.distplot(data_top_street_clear['number'], ax=ax)
ax.set_title('Распределение количества мест в объектах для улиц с большим числом заведений(без выбросов)')


# На графике распределения видно что на улицах Москвы с наибольшим числом заведений преобладают заведения с числом посадочных мест не более 100 шт. 

# <a id="insights"></a>

# #### Выводы:
# 
# - на рынке общественного питания Москвы преобладают кафе, столовые и рестораны;
# - 80,7% объектов несетевые и только 19,3% заведений имеют сетевую структуру;
# - среди сетевых объектов преобладают предприятия быстрого обслуживания и магазины(отделы кулинарии);
# - для сетевых заведений в среднем характерно до 20 объектов с общим числом посадочных мест не более 100 шт;
# - наибольшим числом посадочных мест располагают столовые и рестораны;
# - наиболее популярными районами для дислокации заведений являются Алексеевский район, р-н Коньково, Хорошевский р-н, Пресненский р-н и тд;
# - наименее популярными районами для дислокации заведений являются Таганский р-н, Басманный р-н, р-н Хамовники и тд;
# - для наиболее популярных улиц характерное число посадочных мест в заведениях в среднем не превышает 100 шт.

# <a id="recommendations"></a>

# #### Рекомендации:
# 
# - Таким образом, для открытия заведения в Москве наиболее рекомендуемым видом заведения считаю предприятие быстрого обслуживания или бар, тк конкуренция по данным типам находится на среднем уровне. 
# - Также наиболее рентабельно открывать сразу сеть заведений не более 20 объеков с общим числов посадочных мест от 40 до 100 чел. для хеджирования рисков и увеличения прибыли.
# - Для дислокации объектов предлагаю рассмотреть Кутузовский проспект и Каширское шоссе, тк уровень присутствия в данных районах почти вдвое меньше лидеров, но все же является достаточным для обеспечения рентабельного уровня спроса.

# <a id="presentation"></a>

# Презентация: <https://yadi.sk/i/QOylMXm0nVl6Cw> 

# <div style="border:solid orange 2px; padding: 20px"> <h7 style="color:black; margin-bottom:20px"><font color='orange'> Итог ревью - ты проделала большую работа и главное успешно! Единственное, что дам тебе замечание по поводу выводов - меньше обощени
