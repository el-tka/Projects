#!/usr/bin/env python
# coding: utf-8

# <div style="border:solid green 2px; padding: 20px"> <h1 style="color:green; margin-bottom:20px">Комментарий наставника</h1>
# 
# Привет! Далее в файле мои коммефнтарии ты сможешь найти в ячейках, аналогичных данной ( если рамки комментария зелёные - всё сделано правильно; жёлтые - есть замечания, но не критично; красные - нужно переделать). Не удаляй эти комментарии и постарайся учесть их в ходе выполнения проекта. 

# #### Part 1

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns


# In[2]:


data = pd.read_csv('/datasets/logs_exp.csv', sep = '\t')


# In[3]:


data.info()


# #### Part 2

# In[4]:


data['EventTimestamp'] = pd.to_datetime(data['EventTimestamp'], unit = 's')


# In[5]:


data.head()


# In[6]:


data.columns = ['event_name', 'user_id', 'date_time', 'exp_id']


# In[7]:


data.isnull().sum()


# In[8]:


data['date'] = data['date_time'].dt.date


# In[9]:


data.head()


# #### Part 3

# In[10]:


data['event_name'].value_counts()


# In[11]:


data['event_name'].nunique()


# Всего в данных пять видов событий, наиболее часто встречающееся из которых - "открытие стартовой страницы".

# In[12]:


data['user_id'].nunique()


# Всего в данных 7551 уникальных пользователя.

# In[13]:


event_per_user = data.groupby('user_id')['event_name'].count().reset_index()


# In[14]:


event_per_user['event_name'].describe()


# In[15]:


event_per_user.columns =['user_id', 'event_count']


# In[16]:


np.percentile(event_per_user['event_count'],[95, 99])


# In[17]:


plt.style.use('seaborn')
f, ax = plt.subplots(figsize=(9, 6.5), sharex = True)
sns.despine(left=True)
sns.distplot(event_per_user['event_count'],kde = False, ax=ax)
ax.set_title('Распределение количества событий на одного пользователя')


# В данных определенно присутствуют выбросы, максимальное значение 2308 событий для одного пользователя, однако 75% событий на пользователя не превышает 37 событий, а 95% не превышает 89 событий.

# In[18]:


#уберем выбросы из данных
event_per_user_clear = event_per_user[event_per_user['event_count'] <= 200]


# In[19]:


data = data[data['user_id'].isin(event_per_user_clear['user_id'])]


# In[20]:


data.info()


# In[21]:


plt.style.use('seaborn')
f, ax = plt.subplots(figsize=(9, 6.5), sharex = True)
sns.despine(left=True)
sns.distplot(event_per_user_clear['event_count'], ax=ax)
ax.set_title('Распределение количества событий на одного пользователя (без выбросов)')


# In[22]:


#найдем максимальную и минимальную дату логов
data['date'].describe()


# In[23]:


data['date'].min()


# In[24]:


data['date'].max()


# In[25]:


first_logs = data.groupby('user_id')['date_time'].min().reset_index()


# In[26]:


f, ax = plt.subplots(figsize=(9, 6.5), sharex = True)
plt.hist(first_logs['date_time'], bins=14)
ax.set_title('Гистограмма первых посещений по пользователям')
plt.xticks(rotation = 60)


# In[27]:


f, ax = plt.subplots(figsize=(9, 6.5), sharex = True)
plt.hist(data['date_time'], bins=14)
ax.set_title('Гистограмма по дате и времени')
plt.xticks(rotation = 60)


# In[28]:


data['date'].unique()


# In[29]:


data[data['date']<data['date_time'].dt.date.unique()[6]].groupby('event_name')['user_id'].count()


# In[30]:


data[data['date']<data['date_time'].dt.date.unique()[6]].groupby('event_name')['user_id'].count().sum()


# В данных заметен огромный прилив пользователей 1.08.2019 и последующая увеличившаяся активность пользователей со 2.08.2019 по 8.08.2019. В данных видно что на число пользователей, совершавших действия до 31.07.2019 приходится 524 визита стартовой страницы, 100 просмотров предложений на сайте, 108 добавлений в корзину, 60 оплат и 2 просмотров руководства пользователя. Отбросив данные события мы потеряем 794 шт.

# In[31]:


#сформируем когорты, чтобы проанализировать активность на сайте по датам
first_logs.head()


# In[32]:


first_logs.columns = ['user_id', 'first_date_time']


# In[33]:


data.head()


# In[34]:


data = data.merge(first_logs, on='user_id', how = 'left')


# In[35]:


data['first_date'] = data['first_date_time'].dt.date


# In[36]:


data['date'] = pd.to_datetime(data['date'])
data['first_date'] = pd.to_datetime(data['first_date'])


# In[37]:


data.tail()


# In[38]:


data_grouped_by_cohorts = data.groupby(['first_date', 'date']).agg({'user_id':'nunique'}).reset_index()


# In[39]:


data_grouped_by_cohorts.head()


# In[40]:


data_grouped_by_cohorts['first_date'] = data_grouped_by_cohorts['first_date'].dt.strftime('%m-%d')
data_grouped_by_cohorts['date'] = data_grouped_by_cohorts['date'].dt.strftime('%m-%d')


# In[41]:


data_grouped_by_cohorts_pivot= data_grouped_by_cohorts.pivot_table(index='first_date', columns = ['date'], values='user_id')


# In[42]:


f, ax = plt.subplots(figsize=(13, 9))
sns.heatmap(data_grouped_by_cohorts_pivot, annot=True, fmt='.1f', linewidths=1, linecolor='gray', ax=ax)
plt.yticks(rotation=0)
ax.set_title('Распределение активных пользователей по когортам в разрезе дат')


# На графике видно, что число активных пользователей начинает расти с 31.07. До этой даты пользователи когорт с 25.07 по 30.07 не проявляют существенную активность.

# In[43]:


data_grouped_by_cohorts_events = data.groupby(['first_date', 'event_name']).agg({'user_id':'nunique'}).reset_index()
data_grouped_by_cohorts_events['first_date'] = data_grouped_by_cohorts_events['first_date'].dt.strftime('%m-%d')


# In[44]:


data_grouped_by_cohorts_events_pivot = data_grouped_by_cohorts_events.pivot_table(index='first_date', columns = ['event_name'], values='user_id')


# In[45]:


f, ax = plt.subplots(figsize=(13, 9))
sns.heatmap(data_grouped_by_cohorts_events_pivot, annot=True, fmt='.1f', linewidths=1, linecolor='gray', ax=ax)
plt.yticks(rotation=0)
ax.set_title('Распределение количества событий по когортам')


# На графике видно, что наиболее активной когортой является 1.08. До этой даты пользователи когорт с 25.07 по 30.07 не проявляют существенную активность.

# In[46]:


#таким образом отбросим пользователей из когорт с 25.07 по 30.07
data = data[data['first_date']>data['first_date'].unique()[5]]


# In[47]:


data['first_date'].unique()


# In[48]:


data['exp_id'].value_counts()


# Отбросив данные мы потеряли 794 события. Однако в оставшихся данных присутствуют события из всех экспериментальных групп. Количество событий в группах почти равно между собой.
# 
# <div style="border:solid green 2px; padding: 20px"> <h1 style="color:green; margin-bottom:20px">Комментарий наставника</h1>
# 
# Хорошо, все верно рассчитано. Группы примерно равнозначны - можно переходить к анализу воронки. 

# #### Part 4

# In[49]:


#найдем частоту событий 
data_group_by_event = data['event_name'].value_counts().reset_index()


# In[50]:


#найдем количество уникальных пользователей по каждому событию
users_group_by_event = data.groupby('event_name').agg({'user_id':'nunique'}).sort_values(by = 'user_id',ascending = False).reset_index()


# In[51]:


data_group_by_event = data_group_by_event.merge(users_group_by_event, left_on='index', right_on='event_name')


# In[52]:


data_group_by_event = data_group_by_event[['index', 'event_name_x','user_id']]


# In[53]:


data_group_by_event.columns=['event_name', 'event_amount', 'user_amount']


# In[54]:


#соединим отсортированные данные
data_group_by_event


# In[55]:


#найдем общее количество уникальных пользователей
unique_user_count= data.agg({'user_id':'nunique'})


# In[56]:


unique_user_count[0]


# In[57]:


#найдем долю уникальных пользователей по каждому событию в общем числе уникальных пользвателей
data_group_by_event['user_share']=(data_group_by_event['user_amount']/unique_user_count[0]).round(2)


# In[58]:


data_group_by_event


# In[59]:


f, ax = plt.subplots(figsize=(6,6), sharex=True)
sns.barplot(x=data_group_by_event['event_name'], y=data_group_by_event['user_share'], palette="rocket", ax=ax)
plt.xticks(rotation=30)
ax.set_title('Доля пользователей, совершивших событие хотя бы 1 раз')


# Исходя из полученных данных можно предположить, что цепочка последовательных событий формируется следующим образом:
# 1. Открытие стартовой страницы
# 2. Просмотр предложений на сайте
# 3. Открытие корзины
# 4. Оплата товара
# 
# Просмотр руководства пользователя не входит в данную цепочку, так как может совершаться на любом из этапов.

# In[60]:


#составим простую продуктовую воронку
product_funnel = data_group_by_event[['event_name', 'user_amount']]
product_funnel = product_funnel[product_funnel['event_name']!='Tutorial']


# In[61]:


product_funnel


# In[62]:


#составим продуктовую воронку с учетом последовательности событий
data.head()


# In[63]:


users = data.pivot_table(index='user_id', columns='event_name', values='date_time', aggfunc='min')


# In[64]:


users.head()


# In[65]:


step_1 = ~users['MainScreenAppear'].isna()
step_2 = step_1 & (users['OffersScreenAppear'] > users['MainScreenAppear'])
step_3 = step_2 & (users['CartScreenAppear'] > users['OffersScreenAppear'])
step_4 = step_3 & (users['PaymentScreenSuccessful'] > users['CartScreenAppear'])

n_pageview = users[step_1].shape[0]
n_offerview = users[step_2].shape[0]
n_checkout = users[step_3].shape[0]
n_payment = users[step_4].shape[0]

print('Открытие стартовой страницы:', n_pageview)
print('Просмотр предложений на сайте:', n_offerview)
print('Открытие корзины:', n_checkout)
print('Оплата товара:', n_payment)


# In[66]:


dicta = {'MainScreenAppear': n_pageview, 
        'OffersScreenAppear':n_offerview, 
        'CartScreenAppear':n_checkout,
        'PaymentScreenSuccessful':n_payment}
product_funnel_ordered = pd.DataFrame.from_dict(dicta, orient='index',columns = ['event_amount'])


# In[67]:


product_funnel_ordered = product_funnel_ordered.reset_index()


# In[68]:


product_funnel_ordered.columns = ['event_name', 'user_amount']


# In[69]:


product_funnel_ordered


# In[70]:


from plotly import graph_objects as go
fig = go.Figure()

fig.add_trace(go.Funnel(
    name = 'Простая воронка',
    y = product_funnel['event_name'],
    x = product_funnel['user_amount'],
    textinfo = "value+percent initial",
    opacity = 0.65,
    marker = {"color": "deepskyblue",
              "line": {"width": 4, "color": "wheat"}},
    connector = {"line": {"color": "wheat", "width": 4}}))

fig.add_trace(go.Funnel(
    name = 'Последовательная воронка',
    orientation = "h",
    y = product_funnel['event_name'],
    x = product_funnel_ordered['user_amount'],
    textposition = "inside",
    textinfo = "value+percent initial",
    marker = {"color": "lightsalmon",
              "line": {"width": 4, "color": "wheat"}},
    connector = {"line": {"color": "wheat", "width": 4}}))
fig.update_layout(title_text='Cоотношение простой продуктовой воронки и воронки с учетом последовательности событий')
fig.show()


# На графике видно:
# - что среди пользователей, последовательно совершающих события на сайте, только 6,2% доходят до оплаты. Однако общая доля уникальных пользователей, дошедших до оплаты, составляет 47,8%. Это может быть связано с тем, что из 3263 пользователей, оплативших товары, 2841 пользователь не придерживались предполагаемого потребительского пути. Это скорее всего связано с дополнительным источником трафика, через который пользователи сразу попадают на страницу предложений товаров, минуя стартовую страницу сайта;
# - что среди пользователей, последовательно совершающих события на сайте, больше всего пользователей теряются на шаге оплаты, тк их доля составляет лишь 27,6% от числа пользователей, оформлявших заказ в корзине. Однако среди всех пользователей наибольшие потери наблюдаются при переходе к просмотру предложений, конверсия на этом шаге составляет 62,4%. Тем не менее если предположить дополнительный источник трафика, то на данном шаге добавляются дополнительные 449 пользователей.
# 
# <div style="border:solid green 2px; padding: 20px"> <h1 style="color:green; margin-bottom:20px">Комментарий наставника</h1>
# 
# Воронка построена, проанализирована и визуализирована верно. 

# #### Part 5

# In[71]:


data.head()


# In[72]:


user_amount_by_group = data.groupby('exp_id').agg({'user_id':'nunique'})


# In[73]:


user_amount_by_group


# In[74]:


#сгруппируем данные по группам и дням и посмотрим, есть ли между средними выборок 246 и 247 статистически 
#значимые различия
data_grouped_by_date=data.groupby(['exp_id','date']).agg({'user_id':'nunique'}).reset_index()


# In[75]:


data_grouped_by_date


# In[76]:


import scipy.stats as stats
alpha=0.05


# In[77]:


sample246= data_grouped_by_date[data_grouped_by_date['exp_id']==246]['user_id']


# In[78]:


sample247= data_grouped_by_date[data_grouped_by_date['exp_id']==247]['user_id']


# In[79]:


sample246


# In[80]:


sample247


# In[81]:


results = stats.mannwhitneyu(sample246, sample247)
print("p-value:{0:.3f}".format(results.pvalue))
if (results.pvalue < alpha):
    print("Отвергаем нулевую гипотезу: разница статистически значима")
else:
    print("Не получилось отвергнуть нулевую гипотезу, вывод о различии сделать нельзя")


# По результатам непараметрического теста Манна-Уитни нельзя сделать вывод о структурном различии между количеством уникальных пользователей выборок 246 и 247

# In[82]:


results = stats.ttest_ind(sample246, sample247)
print("p-value:{0:.3f}".format(results.pvalue))
if (results.pvalue < alpha):
    print("Отвергаем нулевую гипотезу: разница статистически значима")
else:
    print("Не получилось отвергнуть нулевую гипотезу, вывод о различии сделать нельзя")


# По результатам проверки t-теста не получилось отвергнуть нулевую гипотезу о равенстве средних выборок 247 и 246, т.е. среднее количество уникальных пользователей по дням статистически равно

# In[83]:


#сгруппируем данные по группам и событиям и посчитаем уникальных пользователей
grouped_by_event246 = data[data['exp_id']==246].groupby('event_name').agg({'user_id':'nunique'}).reset_index()
grouped_by_event247 = data[data['exp_id']==247].groupby('event_name').agg({'user_id':'nunique'}).reset_index()
grouped_by_event248 = data[data['exp_id']==248].groupby('event_name').agg({'user_id':'nunique'}).reset_index()


# In[84]:


grouped_by_event246['share']=grouped_by_event246['user_id']/grouped_by_event246['user_id'].sum()
grouped_by_event247['share']=grouped_by_event247['user_id']/grouped_by_event247['user_id'].sum()
grouped_by_event248['share']=grouped_by_event248['user_id']/grouped_by_event248['user_id'].sum()


# In[85]:


grouped_by_event246


# In[86]:


grouped_by_event247


# Из предыдущих шагов анализа нам известно, что наиболее популярным событием является заход пользователей на главную страницу сайта. Посмотрим, есть ли статичтически значимое разничие между долями захода на сайт в двух контрольных группах

# Т.к. количество необходимых сравнений долей по одним и тем же данным равно 4, применим поправку Шидака. Однако целесообразность данной поправки спорна тк она лишь уменьшает уровень значимости, а достигнутые в ходе эксперимента значения p-value намного превышают 0.05, поэтому поправка на уровень значимости не окажет влияние на результат эксперимента

# In[110]:


alpha = 1 - (1 - alpha)**(1/4)


# In[111]:


alpha


# In[112]:


from scipy import stats as st
import numpy as np
import math as mth


# In[113]:


p1= grouped_by_event246['share'][1]
p2= grouped_by_event247['share'][1]
sum1= grouped_by_event246['user_id'].sum()
sum2= grouped_by_event247['user_id'].sum()


# In[114]:


def z_test(p1, p2, sum1, sum2):
    p_combined = (grouped_by_event246['user_id'][1] + grouped_by_event247['user_id'][1]) /(sum1 +sum2)

    difference = p1 - p2

    z_value = difference / mth.sqrt(p_combined * (1 - p_combined) * (1/sum1 + 1/sum2))

    distr = st.norm(0, 1)
    p_value = (1 - distr.cdf(abs(z_value))) * 2

    print('p-значение: ', p_value)

    if (p_value < alpha):
        return("Отвергаем нулевую гипотезу: между долями есть значимая разница")
    else:
        return("Не получилось отвергнуть нулевую гипотезу, нет оснований считать доли разными")


# <div style="border:solid green 2px; padding: 20px"> <h1 style="color:green; margin-bottom:20px">Комментарий наставника</h1>
# 
# Функция для проверки написана верно.
#     
# Alpha хорошо подобрано

# In[115]:


z_test(p1, p2, sum1, sum2)


# Проверим наличие статистически значимых различий между долями остальных событий

# In[116]:


z_test(grouped_by_event246['share'][0], grouped_by_event247['share'][0], sum1, sum2)
z_test(grouped_by_event246['share'][2], grouped_by_event247['share'][2], sum1, sum2)
z_test(grouped_by_event246['share'][3], grouped_by_event247['share'][3], sum1, sum2)
z_test(grouped_by_event246['share'][4], grouped_by_event247['share'][4], sum1, sum2)


# Для всех остальных событий статистически значимых отличий между долями не обнаружено, поэтому можно сделать вывод, что рабиение на группы произведено корректно.

# In[117]:


sum1 = grouped_by_event246['user_id'].sum()
sum2 = grouped_by_event248['user_id'].sum()


# In[95]:


#сравним результаты группы 248 с результатами контрольных групп 246 и 247
z_test(grouped_by_event246['share'][0], grouped_by_event248['share'][0], sum1, sum2)
z_test(grouped_by_event246['share'][1], grouped_by_event248['share'][1], sum1, sum2)
z_test(grouped_by_event246['share'][2], grouped_by_event248['share'][2], sum1, sum2)
z_test(grouped_by_event246['share'][3], grouped_by_event248['share'][3], sum1, sum2)
z_test(grouped_by_event246['share'][4], grouped_by_event248['share'][4], sum1, sum2)


# In[96]:


sum1 = grouped_by_event247['user_id'].sum()
sum2 = grouped_by_event248['user_id'].sum()


# In[97]:


z_test(grouped_by_event247['share'][0], grouped_by_event248['share'][0], sum1, sum2)
z_test(grouped_by_event247['share'][1], grouped_by_event248['share'][1], sum1, sum2)
z_test(grouped_by_event247['share'][2], grouped_by_event248['share'][2], sum1, sum2)
z_test(grouped_by_event247['share'][3], grouped_by_event248['share'][3], sum1, sum2)
z_test(grouped_by_event247['share'][4], grouped_by_event248['share'][4], sum1, sum2)


# По результатам теста не выявлено статистических различий между долями группы с измененным шрифтом и долями каждой из контрольных групп

# Сравним результаты группы 248 с результатами объединенной контрольной группой, построим таблицу сопряженности для групп 246 и 247

# In[98]:


data.head()


# In[99]:


grouped_by_event_246 = data[data['exp_id']==246]
grouped_by_event_247 = data[data['exp_id']==247]


# In[100]:


#найдем уникальных пользователей, присутствующих только в 246 группе
unique246 = data[np.logical_and(data['user_id'].isin(grouped_by_event_246['user_id']),np.logical_not(data['user_id'].isin(grouped_by_event_247['user_id'])))].groupby('event_name').agg({'user_id':'nunique'})


# In[101]:


#найдем уникальных пользователей, присутствующих только в 247 группе
unique247 = data[np.logical_and(data['user_id'].isin(grouped_by_event_247['user_id']),np.logical_not(data['user_id'].isin(grouped_by_event_246['user_id'])))].groupby('event_name').agg({'user_id':'nunique'})


# In[102]:


unique246


# In[103]:


unique247


# Из таблиц видно что количество уникальных пользователей по группам не изменилось по сравнению со стандартной сортировкой по группам, а значит группы 246 и 247 имеют уникальные субъекты анализа. 

# In[118]:


merged = unique246.merge(unique247, on='event_name')


# In[119]:


merged


# In[120]:


merged['user_count']=merged['user_id_x']+merged['user_id_y']


# In[121]:


merged['share']= merged['user_count']/merged['user_count'].sum()


# In[108]:


sum1 = grouped_by_event248['user_id'].sum()
sum2 = merged['user_count'].sum()


# In[109]:


z_test(grouped_by_event248['share'][0], merged['share'][0], sum1, sum2)
z_test(grouped_by_event248['share'][1], merged['share'][1], sum1, sum2)
z_test(grouped_by_event248['share'][2], merged['share'][2], sum1, sum2)
z_test(grouped_by_event248['share'][3], merged['share'][3], sum1, sum2)
z_test(grouped_by_event248['share'][4], merged['share'][4], sum1, sum2)


# В результате сравнения долей группы 248 с результатами объединенной контрольной группы статистически значимых различий не обнаружено
# 
# <div style="border:solid green 2px; padding: 20px"> <h1 style="color:green; margin-bottom:20px">Комментарий наставника</h1>
# 
# Гипотезы проверены правильно. 

# #### Выводы:
# 1. аб-тестирование производилось по данным с 7551 уникальных пользователей и по 5 событиям за период с 1.08.2019 по 07.08.2020
# 2. на одного пользователя приходится в среднем 20 событий
# 3. 48% всех пользователей доходят до оплаты, однако в приложении существует как минимум 2 источника трафика, в одном из которых пользователи сразу попадают на страницу предложений 
# 4. по результатам различных статистических тестов значимых различий между группами не обнаружено, значит изменение шрифта не оказывает существенного влияния на активность уникальных пользователей
# 
# <div style="border:solid green 2px; padding: 20px"> <h1 style="color:green; margin-bottom:20px">Комментарий наставника</h1>
# 
# Проект выполнен отлично, молодец. 

# In[ ]:




