import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
import calendar
import seaborn as sns
sns.set(style='white', palette='deep')
plt.style.use('grayscale')
warnings.filterwarnings('ignore')
width = 0.35

# Funções
def autolabel(rects,ax, df): #autolabel
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{} ({:.2f}%)'.format(height, height*100/df.shape[0]),
                    xy = (rect.get_x() + rect.get_width()/2, height),
                    xytext= (0,3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=15)
        
def autolabel_without_pct(rects,ax): #autolabel
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy = (rect.get_x() + rect.get_width()/2, height),
                    xytext= (0,3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=15)
def autolabel_horizontal(rects,ax):
    """
    Attach a text label above each bar displaying its height
    """
    for rect in rects:
        width = rect.get_width()
        ax.text(rect.get_x() + rect.get_width()+3, rect.get_y() + rect.get_height()/2.,
                '%.2f' % width,
                ha='center', va='center', color='black', fontsize=15) 

# Importando o Arquivo
df = pd.read_excel('Banco de Dados - WDO.xlsx')

# Verificando Null Values
df.isnull().sum()
null_values = (df.isnull().sum()/len(df)*100)
null_values = pd.DataFrame(null_values, columns= ['% Null Values'])
null_values

# Deletando Null Values
df_feature = df.copy()
df_feature.dropna(inplace=True)
df_feature.isnull().sum()

# Alterando nome de colunas
bank = ['corretora_', 'bank_']
letters = ['abcdefghijlmnopqrstuvxz']
new_columns = np.array([])
for i in bank:
    for j in range(0,4):
        new_columns = np.append(new_columns, i+list(letters[0])[j])
df_feature.columns
count = 0
for i in df_feature.loc[:, ['win_xp_(5m)', 'win_rico_(5m)', 'win_clear_(5m)',
       'win_modal_(5m)', 'win_ubs_(5m)', 'win_btg_(5m)', 'win_bradesco_(5m)',
       'win_genial(5m)']]:
    df_feature.rename(columns={i:new_columns[count]}, inplace=True)
    count+=1

# Verificando erro de digitação
df_feature.columns
df_feature.set_index('data', inplace=True)
max_value = np.array([])
min_value = np.array([])
max_index = np.array([])
min_index = np.array([])
max_time =  np.array([])
min_time =  np.array([])
count = 0
value_error_final = pd.DataFrame()
for i in df_feature.loc[:,['abertura', 'maxima', 'minima',
       'fechamento', '20mma_maxima_2m', '20mma_minima_2m', '9mme_fechamento',
       '200mma_fechamento', '20mma_maxima_5m', '20mma_minima_5m',
       'volume_financeiro','corretora_a', 'corretora_b', 'corretora_c',
       'corretora_d', 'bank_a', 'bank_b', 'bank_c', 'bank_d','gain', 'quantas_correcoes',
       'quantos_pontos_avancou', 'quantos_pontos_retornados']]:
    max_value = np.append(max_value,df_feature[i].max())
    min_value = np.append(min_value,df_feature[i].min())
    max_index = np.append(max_index,df_feature.loc[:,i].idxmax())
    min_index = np.append(min_index,df_feature.loc[:,i].idxmin())
    max_time =  np.append(max_time,df_feature[df_feature[i] == df_feature[i].max()]['horario'])
    min_time =  np.append(min_time,df_feature[df_feature[i] == df_feature[i].min()]['horario'])
    print('O máximo valor para a coluna |{}| foi de {}, no dia {} e no horário {}'.format(i,max_value[count], 
          max_index[count],max_time[count]))
    print('O mínimo valor para a coluna |{}| foi de {}, no dia {} e no horário {}'.format(i,min_value[count], 
          min_index[count], min_time[count]))
    print('*'*100)
    valer_error = pd.DataFrame({'valor_max':[max_value[count]],
                                'dia_max':  [max_index[count]],
                                'horario_max': [max_time[count]],
                                'valor_min':[min_value[count]],
                                'dia_min':  [min_index[count]],
                                'horario_min': [min_time[count]]}, index=[i])
    value_error_final = pd.concat([valer_error,value_error_final])
    count+=1
df_feature = df_feature.drop('gain', axis=1)    


#Pela amplitude podemos verificar erros de digitação nas colunas |máximas| e |mínimas|
df_feature['amplitude'] = df_feature['maxima']-df_feature['minima'] # Criando coluna amplitude
amplitude_error = df_feature[df_feature['amplitude'] <0][['maxima', 'minima', 'horario']]

#Verificando se a ME9 está menor que as MA20 de ativação 
nove_compra_error = df_feature[df_feature['tipo_de_negociacao']=='compra'][['20mma_maxima_2m', '9mme_fechamento'
                              , 'horario']]
nove_venda_error = df_feature[df_feature['tipo_de_negociacao']=='venda'][['20mma_minima_2m', '9mme_fechamento'
                              , 'horario']]
nove_compra_error['error'] = nove_compra_error['9mme_fechamento']-nove_compra_error['20mma_maxima_2m']
nove_venda_error['error'] = nove_venda_error['9mme_fechamento']-nove_venda_error['20mma_minima_2m']
nove_compra_error = nove_compra_error[nove_compra_error['error'].values<0]
nove_venda_error = nove_venda_error[nove_venda_error['error'].values>0]

nove_compra_error[['20mma_maxima_2m', '9mme_fechamento']] = nove_compra_error[['9mme_fechamento','20mma_maxima_2m']].where(nove_compra_error['error']<0,
                 nove_compra_error[['20mma_maxima_2m', '9mme_fechamento']].values)
nove_venda_error[['20mma_minima_2m','9mme_fechamento']] = nove_venda_error[['9mme_fechamento','20mma_minima_2m']].where(nove_venda_error['error']>0,
                 nove_venda_error[['20mma_minima_2m','9mme_fechamento']].values) 

df_feature.groupby(df_feature.index)['horario'].get_group('2019-06-03 ')[0]
df_feature.groupby(df_feature.index)['horario'].get_group('2019-06-19 ')[0] 
df_feature.groupby(df_feature.index)['horario'].value_counts()

for i in range(0, len(nove_compra_error)):
    df_feature.loc[(df_feature.index == nove_compra_error.index[i]) & (df_feature['horario']==nove_compra_error['horario'][i]), '20mma_maxima_2m'] = nove_compra_error['20mma_maxima_2m'].values[i] 
    df_feature.loc[(df_feature.index == nove_compra_error.index[i]) & (df_feature['horario']==nove_compra_error['horario'][i]), '9mme_fechamento'] = nove_compra_error['9mme_fechamento'].values[i]
for i in range(0, len(nove_venda_error)):
    df_feature.loc[(df_feature.index == nove_venda_error.index[i]) & (df_feature['horario']==nove_venda_error['horario'][i]), '20mma_minima_2m'] = nove_venda_error['20mma_minima_2m'].values[i] 
    df_feature.loc[(df_feature.index == nove_venda_error.index[i]) & (df_feature['horario']==nove_venda_error['horario'][i]), '9mme_fechamento'] = nove_venda_error['9mme_fechamento'].values[i]

nove_venda_error['20mma_minima_2m'][1]
df_feature.loc[(df_feature.index == nove_venda_error.index[1]) & (df_feature['horario']==nove_venda_error['horario'][1]), '20mma_minima_2m'] 

#Verificando se M20 2m high tem divergência com M20 2m Low
df_feature.columns
m20_error_high_2m = df_feature[df_feature['20mma_maxima_2m']<df_feature['20mma_minima_2m']][['20mma_maxima_2m', '20mma_minima_2m', 'horario']]
m20_error_high_2m['error'] = m20_error_high_2m['20mma_maxima_2m']-m20_error_high_2m['20mma_minima_2m']
m20_error_high_2m[['20mma_maxima_2m', '20mma_minima_2m']] = m20_error_high_2m[['20mma_minima_2m', '20mma_maxima_2m']].where(m20_error_high_2m['error']<0,
                 m20_error_high_2m[['20mma_maxima_2m', '20mma_minima_2m']].values)
for i in range(0, len(m20_error_high_2m)):
    df_feature.loc[(df_feature.index == m20_error_high_2m.index[i]) & (df_feature['horario']==m20_error_high_2m['horario'][i]), '20mma_maxima_2m'] = m20_error_high_2m['20mma_maxima_2m'].values[i] 
    df_feature.loc[(df_feature.index == m20_error_high_2m.index[i]) & (df_feature['horario']==m20_error_high_2m['horario'][i]), '20mma_minima_2m'] = m20_error_high_2m['20mma_minima_2m'].values[i] 

#Verificando se M20 5m high tem divergência com M20 5m Low
df_feature.columns
m20_error_high_5m = df_feature[df_feature['20mma_maxima_5m']<df_feature['20mma_minima_5m']][['20mma_maxima_5m', '20mma_minima_5m', 'horario']]
m20_error_high_5m['error'] = m20_error_high_5m['20mma_maxima_5m']-m20_error_high_5m['20mma_minima_5m']
m20_error_high_5m[['20mma_maxima_5m', '20mma_minima_5m']] = m20_error_high_5m[['20mma_minima_5m', '20mma_maxima_5m']].where(m20_error_high_5m['error']<0,
                 m20_error_high_5m[['20mma_maxima_5m', '20mma_minima_5m']].values)
for i in range(0, len(m20_error_high_5m)):
    df_feature.loc[(df_feature.index == m20_error_high_5m.index[i]) & (df_feature['horario']==m20_error_high_5m['horario'][i]), '20mma_maxima_5m'] = m20_error_high_5m['20mma_maxima_5m'].values[i] 
    df_feature.loc[(df_feature.index == m20_error_high_5m.index[i]) & (df_feature['horario']==m20_error_high_5m['horario'][i]), '20mma_minima_5m'] = m20_error_high_5m['20mma_minima_5m'].values[i]

#Salvando planilha tratada
df_feature.to_excel('WDO Tratado.xlsx')

#Quais foram as operações com maior frequência? PLOT
df_feature.columns
df_compra = df_feature[df_feature['tipo_de_negociacao']=='compra']['tipo_de_negociacao']
df_venda = df_feature[df_feature['tipo_de_negociacao']=='venda']['tipo_de_negociacao']
labels = [df_compra.values[0],df_venda.values[0]]
ind = np.arange(len(labels))
values = [len(df_compra), len(df_venda)]
fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(1,1,1)
ax.set_title('Quantidade de Operações de Compra e Veda \n Realizado pela Estratégia de negociação', fontsize=15)
ax.set_xlabel('Tipo de Operação', fontsize=15)
ax.set_ylabel('Quantidade de Operações Realizadas', fontsize=15)
ax.set_xticklabels(['Compra', 'Venda'], fontsize=15)
ax.set_yticklabels(np.arange(0,501,100), fontsize=15)
rects1= ax.bar('Compra', len(df_compra), width, edgecolor='black')
rects2=ax.bar('Venda', len(df_venda), width, edgecolor='black')
ax.set_xticks(ind)
autolabel(rects1,ax,df_feature)
autolabel(rects2,ax,df_feature)
plt.tight_layout()

#Quais foram os dias com maiores operações? PLOT
df_feature.columns
df_compra = df_feature[df_feature['tipo_de_negociacao']=='compra'][['horario','tipo_de_negociacao']]
df_venda = df_feature[df_feature['tipo_de_negociacao']=='venda'][['horario','tipo_de_negociacao']]
df_compra['data'] = df_compra.index
df_compra['dia'] = df_compra['data'].apply(lambda x: x.weekday())
df_compra['mes'] = df_compra['data'].apply(lambda x: x.month)
df_compra['hora'] = df_compra['horario'].apply(lambda x: x.hour)

df_venda['data'] = df_venda.index
df_venda['dia'] = df_venda['data'].apply(lambda x: x.weekday())
df_venda['mes'] = df_venda['data'].apply(lambda x: x.month)
df_venda['hora'] = df_venda['horario'].apply(lambda x: x.hour)

dias = {}
for i,v in enumerate(list(calendar.day_name)):
    dias[i]=v
    
meses = {}
for i,v in enumerate(list(calendar.month_name)[1:],1):
    meses[i]=v
    
dias_nomes_compra = np.array([])
for i in df_compra['dia']:
    for j in range(0,len(dias)):
        if i == list(dias.keys())[j]:
            dias_nomes_compra = np.append(dias_nomes_compra,dias[j])
            
dias_nomes_venda = np.array([])
for i in df_venda['dia']:
    for j in range(0,len(dias)):
        if i == list(dias.keys())[j]:
            dias_nomes_venda = np.append(dias_nomes_venda,dias[j])
            
def compra_venda(x):
    for i in range(6,len(meses)+1):
        if x == i:
            return meses[x]
        
df_compra['dia'] = dias_nomes_compra
df_venda['dia'] = dias_nomes_venda
df_compra['mes'] = df_compra['mes'].apply(compra_venda )
df_venda['mes'] = df_venda['mes'].apply(compra_venda )

fig = plt.figure(figsize=(15,15))
ax = fig.add_subplot(1,1,1)
labels = np.array([])
for i in range(0,5):
    labels = np.append(labels,dias[i]) 
len_dia_compra = np.array([])
len_dia_venda = np.array([])
for i in labels:
    len_dia_compra = np.append(len_dia_compra, len(df_compra[df_compra['dia']==i]))
    len_dia_venda = np.append(len_dia_venda, len(df_venda[df_venda['dia']==i]))        
ind = np.arange(len(labels))
ax.set_title('Tabela de Operações por Dias', fontsize=15)
ax.set_xticks(ind)
ax.set_xticklabels(['Segunda', 'Terça', 'Quarta', 'Quinta', 'Sexta'], fontsize=15)
ax.set_xlabel('Dias da semana', fontsize=15)
ax.set_yticklabels(np.arange(0,150,20),fontsize=15)
ax.set_ylabel('Quantidade de Operações por Dias', fontsize=15)
for i in range(0,len(labels)):  
    rects1 = ax.bar(ind+width/2, len_dia_compra, width=width, edgecolor='black')
    rects2 = ax.bar(ind-width/2, len_dia_venda, width=width, edgecolor='black')
ax.legend(['compra','venda' ], fontsize=15, loc='best')    
autolabel_without_pct(rects1,ax)
autolabel_without_pct(rects2,ax)
plt.tight_layout()

#Quais foram os mesmos com maires operações? PLOT
labels = []
len_mes_compra = []
len_mes_venda = []
[labels.append(meses[i]) for i in range(6,13)]
[len_mes_compra.append(len(df_compra[df_compra['mes']==i])) for i in labels]
[len_mes_venda.append(len(df_venda[df_venda['mes']==i])) for i in labels]
ind=np.arange(len(labels))

fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(1,1,1)
ax.set_title('Tabela de Operações por Mês', fontsize=15)
ax.set_xticks(ind)
ax.set_xticklabels(['Junho', 'Julho', 'Agosto', 'Setembro', 'Outubro', 'Novembro', 'Dezembro'], fontsize=15)
ax.set_yticklabels(np.arange(0,110,10), fontsize=15)
ax.set_ylabel('Quantidade de Operações por Mês', fontsize=15)
for i in range(0,len(len_mes_compra)):
    rects1= ax.bar(ind+width/2, len_mes_compra, width=width, edgecolor='black')
    rects2= ax.bar(ind-width/2, len_mes_venda, width=width, edgecolor='black')
ax.legend(['compra','venda' ], fontsize=15, loc='best')
autolabel_without_pct(rects1,ax,)
autolabel_without_pct(rects2,ax,)
plt.tight_layout()

#Quais horários obteve mais sinais? PLOT
bins = np.arange(9,18)
time = list(np.arange(9,18))
time_string = [str(time[i]) for i in range(0,len(time))]

len_time_compra = list(df_compra.groupby(pd.cut(df_compra['hora'], bins))['hora'].value_counts().values)
len_time_venda = list(df_venda.groupby(pd.cut(df_venda['hora'], bins))['hora'].value_counts().values)

labels=[]
count1=0
count2=1
while count2 != len(time_string):
    labels.append('['+time_string[count1]+' - '+time_string[count2]+']')
    count1+=1
    count2+=1
ind = np.arange(len(labels))
fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(1,1,1)
ax.set_title('Tabela de Operações por Horário', fontsize=15)
ax.set_xticklabels(labels, fontsize=15)
ax.set_xticks(ind)
ax.set_xlabel('Horários', fontsize=15)
ax.set_yticklabels(np.arange(0,100,10), fontsize=15)
ax.set_ylabel('Quantidade de Operações por Horário', fontsize=15)
ax.set_yticks(np.arange(0,100,10))
for i in range(0,len(len_time_compra)):
    rects1 = ax.bar(ind+width/2, len_time_compra, width=width, edgecolor='black', color='black')
    rects2 = ax.bar(ind-width/2, len_time_venda, width=width, edgecolor='black', color='gray')
ax.legend(['compra','venda' ],loc='best', fontsize=15)
autolabel_without_pct(rects1,ax)
autolabel_without_pct(rects2,ax)
plt.tight_layout()

#Quantos pontos avançaram?. PLOT
estatistica = df_feature.describe()
df_feature.columns
bins = np.arange(0,52,4)
df_feature.groupby(pd.cut(df_feature['quantos_pontos_avancou'], bins))['tipo_de_negociacao'].value_counts()

len_pontos_compra = []
len_pontos_venda = []
for i in range(0, 12):
    grouped = df_feature.groupby(pd.cut(df_feature['quantos_pontos_avancou'], bins))['tipo_de_negociacao'].value_counts().index.levels[0][i]
    grouped_values = df_feature.groupby(pd.cut(df_feature['quantos_pontos_avancou'], bins))['tipo_de_negociacao'].get_group(grouped).value_counts()
    if grouped_values.index[0] == 'compra':
        len_pontos_compra.append(grouped_values.values[0])
    if grouped_values.index[1] == 'compra':
        len_pontos_compra.append(grouped_values.values[1])
    if grouped_values.index[0] == 'venda':
        len_pontos_venda.append(grouped_values.values[0])
    if grouped_values.index[1] == 'venda':
        len_pontos_venda.append(grouped_values.values[1])
pontos = np.arange(0,49,4)
pontos_string = [str(i) for i in pontos]
count=0
count1=1
labels = []
while count1 != len(pontos_string):
    labels.append('['+pontos_string[count]+'-'+pontos_string[count1]+']')
    count+=1
    count1+=1

ind =np.arange(len(labels))
fig=plt.figure(figsize=(15,15))
ax1 = fig.add_subplot(2,1,1)
ax2 = fig.add_subplot(2,1,2)

ax1.set_title('Gráfico de Pontos Avançados por \nOperação de Compra e Venda', fontsize=15)
ax1.set_yticklabels(np.arange(0,200,10), fontsize=15)
ax1.set_yticks(np.arange(0,200,10))
ax1.set_ylim(0, 200)
ax1.set_ylabel('Contagem', fontsize=15)
ax1.set_xticklabels(labels, fontsize=15)
ax1.set_xticks(ind)
ax1.set_xlabel('Pontos Avançados', fontsize=15)
rects1 = ax1.bar(ind+width/2,len_pontos_compra, width=width, edgecolor='black', alpha=0.8, label='Operação de Compra')
rects2 = ax1.bar(ind-width/2,len_pontos_venda, width=width, edgecolor='black', alpha=0.8, label ='Operação de Venda')
autolabel_without_pct(rects1,ax1)
autolabel_without_pct(rects2,ax1)
ax1.legend(loc='best',frameon=False )
plt.tight_layout()
 
ax2.hist(df_feature['quantos_pontos_avancou'].values, bins=50,
         density=False, histtype='barstacked', align='mid', color='black', 
         alpha=0.5, range=(0,70))  
ax2.set_ylabel('Frequência', fontsize=15)  
ax2.tick_params(axis='both', labelsize=15, labelcolor='k', labelrotation=0)
ax2.set_xlabel('Pontos', fontsize=15)   
ax2.set_title('Distribuição dos Pontos Avançados', fontsize=15)

#Média por grupo de pontos. PLOT
df_compra['quantos_pontos_avancou'] = df_feature[df_feature['tipo_de_negociacao']=='compra']['quantos_pontos_avancou']
df_venda['quantos_pontos_avancou'] = df_feature[df_feature['tipo_de_negociacao']=='venda']['quantos_pontos_avancou']

mean_pontos_compra = np.round(list(df_compra.groupby(pd.cut(df_compra['quantos_pontos_avancou'], bins))['quantos_pontos_avancou'].mean().values),decimals=2 )
mean_pontos_venda = np.round(list(df_venda.groupby(pd.cut(df_venda['quantos_pontos_avancou'], bins))['quantos_pontos_avancou'].mean().values),decimals=2 )
std_pontos_compra = np.round(list(df_compra.groupby(pd.cut(df_compra['quantos_pontos_avancou'], bins))['quantos_pontos_avancou'].std().values),decimals=2 )
std_pontos_venda = np.round(list(df_venda.groupby(pd.cut(df_venda['quantos_pontos_avancou'], bins))['quantos_pontos_avancou'].std().values),decimals=2 )

fig = plt.figure(figsize=(15,15))
ax = fig.add_subplot(1,1,1)
ax.set_title('Média por Grupo dos Pontos Avançados', fontsize=15)
ax.set_yticklabels(labels, fontsize=15)
ax.set_yticks(ind)
ax.set_ylabel('Grupo de Pontos Avançados', fontsize=15)
ax.set_xlabel('Média por Grupo de Pontos Avançados', fontsize=15)
ax.set_xticklabels(np.arange(0,70,10), fontsize=15)
ax.set_xticks(np.arange(0,70,10))
ax.set_xlim(0,60)
rects1 = ax.barh(ind+width/2,mean_pontos_compra, width, edgecolor='black',align='center', alpha=0.5, label='Operação de Compra', xerr=std_pontos_compra)
rects2 = ax.barh(ind-width/2,mean_pontos_venda, width, edgecolor='black', align='center', alpha=0.5, label='Operação de Venda', xerr=std_pontos_venda)
ax.legend(loc='best', frameon=False, fontsize=15)
autolabel_horizontal(rects1,ax)
autolabel_horizontal(rects2,ax)

#Analisando a maior porcentagem de acertos
pontos_avancados = list(np.arange(1,max(df_feature['quantos_pontos_avancou']),0.5))
for i in np.arange(0.5,len(pontos_avancados),0.5):
    soma = sum(df_feature['quantos_pontos_avancou'].apply(lambda x: x>i))
    print('Acimda de {:.2f} pontos avançou {:.2f}%'.format(i,(soma/len(df_feature))*100))

#Analisando diferença de máxima e mínima
estatistica.columns
estatistica.loc['max',['abertura', 'maxima', 'minima', 'fechamento']] - estatistica.loc['min',['abertura', 'maxima', 'minima', 'fechamento']]

#Analisando média dos preços
estatistica.columns
labels = ['abertura', 'maxima', 'minima', 'fechamento', '20mma_maxima_2m',
       '20mma_minima_2m', '9mme_fechamento', '200mma_fechamento',
       '20mma_maxima_5m', '20mma_minima_5m']
media_colunas = np.round(list(estatistica.loc['mean',labels]), decimals=2)
ind = np.arange(len(labels))
fig = plt.figure(figsize=(15,15)) 
ax = fig.add_subplot(1,1,1)
for i in range(0,len(labels)):
    rects0 = ax.bar(labels[i], media_colunas[i],width=width, edgecolor='black')
    autolabel_without_pct(rects0,ax)
ax.set_xticks(ind)
labels = [labels[i].capitalize() for i in range(0,len(labels))]
ax.set_xticklabels(labels, fontsize=15)
ax.set_title('Media de Pontos por Colunas', fontsize=15)
ax.set_xlabel('Colunas', fontsize=15)
ax.set_ylabel('Média das Coluna', fontsize=15)
ax.set_yticks(np.arange(0,5000,500))
ax.set_yticklabels(np.arange(0,5000,500), fontsize=15)
ax.tick_params(axis='x', labelsize=15, labelcolor='k', labelrotation=45)
plt.tight_layout()

#Definindo a coluna gain
df_feature.columns
df_feature['gain'] = df_feature['quantos_pontos_avancou'].apply(lambda x: x>=2.5)
df_feature['gain'] = df_feature['gain'].apply(lambda x: 1 if x==True else 0)
gain = df_feature[df_feature['gain'] ==1]
loss = df_feature[df_feature['gain'] ==0]
print('Total de {:.2f} operações com Gain'.format(len(gain)))
print('Porcentagem de {:.2f}% gain em relação ao banco de dados'.format((len(gain)/len(df_feature))*100))
print('Total de {:.2f} operações com Loss'.format(len(loss)))
print('Porcentagem de {:.2f}% loss em relação ao banco de dados'.format((len(loss)/len(df_feature))*100))

####################################################################################################
#Definindo os principais bancos e corretoras a favor da operação
df_feature.columns
df_compra = df_feature[df_feature['tipo_de_negociacao']=='compra'][['corretora_a', 'corretora_b', 'corretora_c',
       'corretora_d', 'bank_a', 'bank_b', 'bank_c', 'bank_d','gain']]
df_venda = df_feature[df_feature['tipo_de_negociacao']=='venda'][['corretora_a', 'corretora_b', 'corretora_c',
       'corretora_d', 'bank_a', 'bank_b', 'bank_c', 'bank_d','gain']]
df_compra_gain = df_compra[df_compra['gain']==1]
df_venda_gain = df_venda[df_venda['gain']==1]


bancos = ['corretora_a', 'corretora_b', 'corretora_c',
       'corretora_d', 'bank_a', 'bank_b', 'bank_c', 'bank_d']
df_compra_gain_porcentagem = pd.DataFrame(np.round([(len(df_compra_gain[df_compra_gain[i]==1])/len(df_compra_gain))*100 for i in bancos],decimals=2), 
                                           index= [i for i in bancos], columns=['% Gain'])
df_venda_gain_porcentagem = pd.DataFrame(np.round([(len(df_venda_gain[df_venda_gain[i]==1])/len(df_venda_gain))*100 for i in bancos], decimals=2),
                                          index = [i for i in bancos], columns=['% Loss'])

fig = plt.figure(figsize=(10,10))
ind=np.arange(len(bancos))
ax = fig.add_subplot(1,1,1)
for i in range(len(bancos)):
    rects = ax.bar(bancos[i],df_compra_gain_porcentagem.values[i][0], width=width, edgecolor='black')
    autolabel_without_pct(rects, ax )
    plt.tight_layout()
ax.set_xticklabels(['Corretora A', 'Corretora B', 'Corretora C', 'Corretora D','Banco A', 'Banco B','Banco C','Banco D']
, fontsize=15)#[bancos[i].capitalize() for i in range(len(bancos))]
ax.set_xlabel('Instituições Financeiras', fontsize=15)
ax.tick_params(axis='x', labelsize=15, labelcolor='k', labelrotation=45)
ax.set_xticks(ind)
ax.set_ylabel('Porcentagem', fontsize=15)
ax.set_yticklabels(np.arange(0,70,10), fontsize=15)
ax.set_title('Gráfico de Porcentagem de \n Acertos Instituições Financeiras na Compra', fontsize=15)

fig = plt.figure(figsize=(10,10))
ind=np.arange(len(bancos))
ax = fig.add_subplot(1,1,1)
for i in range(len(bancos)):
    rects = ax.bar(bancos[i],df_venda_gain_porcentagem.values[i][0], width=width, edgecolor='black')
    autolabel_without_pct(rects, ax)
    plt.tight_layout()
ax.set_xticklabels(['Corretora A', 'Corretora B', 'Corretora C', 'Corretora D','Banco A', 'Banco B','Banco C','Banco D']
, fontsize=15)#[bancos[i].capitalize() for i in range(len(bancos))]
ax.set_xlabel('Instituições Financeiras', fontsize=15)
ax.tick_params(axis='x', labelsize=15, labelcolor='k', labelrotation=45)
ax.set_xticks(ind)
ax.set_ylabel('Porcentagem', fontsize=15)
ax.set_yticklabels(np.arange(0,70,10), fontsize=15)
ax.set_title('Gráfico de Porcentagem de \n Acertos das Instituições Financeiras na Venda', fontsize=15)

#Correlação entre as variáveis dependentes e independentes
df_feature.columns
df2 = df_feature.drop(['tipo_de_negociacao','percentual_venda',
                       'quantas_correcoes','quantos_pontos_avancou', 'quantos_pontos_retornados', 'gain'], axis=1)
correlacao = df2.corrwith(df_feature.gain)
labels = [i.capitalize() for i in correlacao.index]
ind = np.arange(len(labels))
fig= plt.figure(figsize=(10,10))
ax=fig.add_subplot(1,1,1)
for i in range(len(labels)):
    rects = ax.bar(labels[i], np.round(correlacao.values[i], decimals=2), width=width, edgecolor='black')
ax.axhline(y=0, color='r', linestyle='-', linewidth =1)
ax.tick_params(axis='x', labelsize=15, labelcolor='k', labelrotation=90)
ax.set_xlabel('Intituições Financeiras', fontsize=15)
ax.set_xticks(ind)
ax.set_ylabel('Correlação', fontsize=15)
ax.tick_params(axis='y', labelsize=15, labelcolor='k', labelrotation=0)
ax.set_ylim(-1,1)
ax.set_title('Correlação da Variáveis Independens \n com a Variável dependente Gain', fontsize=15)

#Separando em X e y 
df_feature.columns
X = df_feature.drop(['horario','percentual_venda',
                     'quantas_correcoes', 'quantos_pontos_avancou', 
                     'quantos_pontos_retornados', 'gain'],axis=1)
y = df_feature['gain']

#Transformando as variáveis Dummies
X = pd.get_dummies(X)

#Evitando dummies trap
X.columns
X = X.drop('tipo_de_negociacao_venda',axis=1)

#Splitting train and test set
from sklearn.model_selection import train_test_split 
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3, random_state=0)
X_train.shape, X_test.shape,y_train.shape, y_test.shape
len(y_train[y_train==1])
len(y_train[y_train==0])

# Balancing the Training Set Upsample
from sklearn.utils import resample

X = pd.concat([X_train, y_train], axis=1)

not_gain = X[X.gain==0]
gain = X[X.gain==1]

gain_upsampled = resample(not_gain,
                          replace=True, # sample with replacement
                          n_samples=len(gain), # match number in majority class
                          random_state=0) # reproducible results

upsampled = pd.concat([gain, gain_upsampled])

upsampled.gain.value_counts()

y_train = upsampled.gain
X_train = upsampled.drop('gain', axis=1)
len(y_train[y_train==1])
len(y_train[y_train==0])

#Normalizando os dados
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
X_train = pd.DataFrame(sc_x.fit_transform(X_train), columns=X_train.columns.values)
X_test = pd.DataFrame(sc_x.transform(X_test), columns=X_train.columns.values)

#Importing Keras libraries e packages
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Activation
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
from keras.layers import LeakyReLU
leaky_relu_alpha = 0.1
import time
from keras.optimizers import Adam, Adamax, Nadam, SGD
from keras import regularizers

#How many layer and neurons I will use in my model?
'''
def create_model(layers,activation,optimizer):
    classifier = Sequential()
    for i, nodes in enumerate(layers):
        if i==0:
            classifier.add(Dense(nodes,input_dim = int(X_train.shape[1])))
            classifier.add(Activation(activation))
        else:
            classifier.add(Dense(nodes))
            classifier.add(Activation(activation))
    classifier.add(Dense(1)) #Note: no activation beyond this point.
    classifier.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    return classifier
    
classifier = KerasClassifier(build_fn=create_model, batch_size = 10, epochs = 100, verbose=0 )
classifier

layers = [[24,48], [48,24],[22,44],[32,64]]
activation = ['sigmoid', 'relu']
parameters = {'batch_size': [128, 256],
              'layers' : layers,
              'activation' : activation,
              'epochs': [100, 500],
              'optimizer': ['adam', 'sgd']}

grid_search = GridSearchCV(estimator = classifier,
                           param_grid = parameters,
                           scoring = 'accuracy',
                           cv = 10)
t0 = time.time()
grid_search = grid_search.fit(X_train, y_train)
t1 = time.time()
print("Took %0.2f seconds" % (t1 - t0))

best_parameters = grid_search.best_params_
best_accuracy = grid_search.best_score_
best_accuracy, best_parameters    
'''
def create_model(layers,dropout):
    classifier = Sequential()
    for i, nodes in enumerate(layers):
        if i==0:
            classifier.add(Dense(nodes,init = 'uniform', activation='relu', input_dim = int(X_train.shape[1])))
            classifier.add(Dropout(p= dropout))
        else:
            classifier.add(Dense(nodes, init = 'uniform'))
            classifier.add(LeakyReLU(alpha=leaky_relu_alpha))
            classifier.add(Dropout(p= dropout ))
            
    classifier.add(Dense(1,init = 'uniform', activation = 'sigmoid')) #Note: no activation beyond this point.
    classifier.compile(optimizer='sgd', loss='binary_crossentropy', metrics=['accuracy'])
    return classifier

classifier = KerasClassifier(build_fn=create_model, batch_size = 10, epochs = 100, verbose=0 )
classifier
int((X_test.shape[1]/2)+1)

layers = [[int((X_test.shape[1]/2)+1),24,48], [int((X_test.shape[1]/2)+1),48,24],[int((X_test.shape[1]/2)+1),24],[int((X_test.shape[1]/2)+1),48]]
dropout = [0.2,0.3,0.4,0.5]
parameters = {
              'layers' : layers,
              'dropout': dropout,
              
              }
#'batch_size': [128, 256],
#'epochs': [100, 500]
grid_search = GridSearchCV(estimator = classifier,
                           param_grid = parameters,
                           scoring = 'accuracy',
                           cv = 10)
t0 = time.time()
grid_search = grid_search.fit(X_train, y_train)
t1 = time.time()
print("Took %0.2f seconds" % (t1 - t0))

best_parameters = grid_search.best_params_
best_accuracy = grid_search.best_score_
best_accuracy, best_parameters

results2 = pd.DataFrame()
for layer1 in np.arange(len(dropout)):
    for layer2 in np.arange(len(dropout)):
        for layer3 in np.arange(len(dropout)):
#Initialising the ANN
            classifier = Sequential()

#Adding the input layer and the first hidden layer
            classifier.add(Dense(output_dim =int((X_test.shape[1]/2)+1) , init = 'uniform', 
                                 activation = 'relu', input_dim = int(X_test.shape[1])))
#classifier.add(LeakyReLU(alpha=leaky_relu_alpha))
            classifier.add(Dropout(p= dropout[layer1] ))
#output_dim = média da soma do número de variáveis (no caso 30) + 1
#init = Peso. Gera aleatoriamente. Sempre 'uniform'
#activation = função de ativação linear retificada. A mais utilizada.
#Input_dim = número de variáveis (no caso 30)
#Dropout = evita overfitting. Começa com 0.1. Se continuar com overfitting, tentar de 0.2 a 0.5. Nunca maior que 0.5 senão será underfitting

#Adding the second hidden layer
            classifier.add(Dense(output_dim = 24, init = 'uniform'))#activation = 'relu'
            classifier.add(LeakyReLU(alpha=leaky_relu_alpha))
            classifier.add(Dropout(p= dropout[layer2]))
#output_dim = média da soma do número de variáveis (no caso 30) + 1
#init = Peso. Gera aleatoriamente. Sempre 'uniform'
#activation = função de ativação linear retificada. A mais utilizada.
#Input_dim = Não será mais necessário pq já foi feito no input layer
#Dropout = evita overfitting. Começa com 0.1. Se continuar com overfitting, tentar de 0.2 a 0.5. Nunca maior que 0.5 senão será underfitting

#Adding the third hidden layer
            classifier.add(Dense(output_dim = 48, init = 'uniform'))#, activation = 'relu'
            classifier.add(LeakyReLU(alpha=leaky_relu_alpha))
            classifier.add(Dropout(p= dropout[layer3]))

#Adding the output layer
            classifier.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid'))
#output_dim = Como neste caso queremos 1 ou 0, só teremos 1 output_layer. 
#init = Peso. Gera aleatoriamente. Sempre 'uniform'
#activation = função de ativação sigmoid. A mais utilizada para output layer quando são binárias. 
#Input_dim = Não será mais necessário pq já foi feito no input layer

#Compiling the ANN
            adam=Adam(lr=0.0001)
            classifier.compile(optimizer = adam, loss = 'binary_crossentropy', metrics = ['accuracy'])
#optimizer = algorítmo para selecionar o melhor peso do ANN. 'Adam' é um dos mais utilizados
#loss =  Algorítmo que minimiza as perdas do gradiente descendete estocástico. Como a saida é binária, utilizou binary_crossentropy. Se houver mais que uma variável categorical_crossentropy
#metrics = Padrão

#Fit classifier to the training test
            history = classifier.fit(X_train, y_train, batch_size = len(X_train), epochs = 100, validation_data=(X_test, y_test))
#batch_size = não tem um valor certo. 
#epochs  = não tem um valor certo

#Predicting the test set result
            y_pred = classifier.predict(X_test)
            y_pred = (y_pred > 0.5) #converte em verdadeiro ou falso

            from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, precision_score, recall_score
            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred)
            rec = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)

            results_final = pd.DataFrame([['ANN Dropout {} - {} - {}'.format(dropout[layer1],dropout[layer2],dropout[layer3]), acc, prec, rec, f1]],
                                     columns = ['Model', 'Accuracy', 'Precision', 'Recall', 'F1 Score'])
            results2 = results2.append(results_final)
            
            # Plot training & validation accuracy values
            plt.plot(history.history['acc'])
            plt.plot(history.history['val_acc'])
            plt.title('Acurácia c/ Dropout {} - {} - {}'.format(dropout[layer1],dropout[layer2],dropout[layer3]))
            plt.ylabel('Acurácia')
            plt.xlabel('Epoch')
            plt.legend(['Treino', 'Teste'], loc='upper left')
            plt.savefig('Accuracy {} - {} - {}.png'.format(dropout[layer1],dropout[layer2],dropout[layer3]))
            plt.show()
            
            # Plot training & validation loss values
            plt.plot(history.history['loss'])
            plt.plot(history.history['val_loss'])
            plt.title('Perda c/ Dropout {} - {} - {}'.format(dropout[layer1],dropout[layer2],dropout[layer3]))
            plt.ylabel('Perda')
            plt.xlabel('Epoch')
            plt.legend(['Treino', 'Teste'], loc='upper left')
            plt.savefig('Loss {} - {} - {}.png'.format(dropout[layer1],dropout[layer2],dropout[layer3]))
            plt.show() 
            
results2.sort_values(by='Accuracy', ascending=False)   
#Salvando planilha tratada
results2.sort_values(by='Accuracy', ascending=False).to_excel('Resltado Final.xlsx')                
###########################################################################################################################            
#Initialising the ANN
classifier = Sequential()

#Adding the input layer and the first hidden layer
classifier.add(Dense(output_dim =int((X_test.shape[1]/2)+1) , init = 'uniform',  input_dim = int(X_test.shape[1]),
                     activation = 'relu'))#kernel_regularizer=regularizers.l1(0.001)
classifier.add(Dropout(p= 0.5))
#classifier.add(LeakyReLU(alpha=leaky_relu_alpha))

#output_dim = média da soma do número de variáveis (no caso 30) + 1
#init = Peso. Gera aleatoriamente. Sempre 'uniform'
#activation = função de ativação linear retificada. A mais utilizada.
#Input_dim = número de variáveis (no caso 30)
#Dropout = evita overfitting. Começa com 0.1. Se continuar com overfitting, tentar de 0.2 a 0.5. Nunca maior que 0.5 senão será underfitting

#Adding the second hidden layer
classifier.add(Dense(output_dim = 24, init = 'uniform',
                     ))#activation = 'relu', kernel_regularizer=regularizers.l1(0.001)
classifier.add(LeakyReLU(alpha=leaky_relu_alpha))
classifier.add(Dropout(p= 0.5))

#output_dim = média da soma do número de variáveis (no caso 30) + 1
#init = Peso. Gera aleatoriamente. Sempre 'uniform'
#activation = função de ativação linear retificada. A mais utilizada.
#Input_dim = Não será mais necessário pq já foi feito no input layer
#Dropout = evita overfitting. Começa com 0.1. Se continuar com overfitting, tentar de 0.2 a 0.5. Nunca maior que 0.5 senão será underfitting

#Adding the third hidden layer
classifier.add(Dense(output_dim = 48, init = 'uniform',
                     ))#, activation = 'relu', kernel_regularizer=regularizers.l1(0.001)
classifier.add(LeakyReLU(alpha=leaky_relu_alpha))
classifier.add(Dropout(p= 0.5))

#Adding the output layer
classifier.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid'))
#output_dim = Como neste caso queremos 1 ou 0, só teremos 1 output_layer. 
#init = Peso. Gera aleatoriamente. Sempre 'uniform'
#activation = função de ativação sigmoid. A mais utilizada para output layer quando são binárias. 
#Input_dim = Não será mais necessário pq já foi feito no input layer

#Compiling the ANN
sgd = SGD(lr=0.0001)
adam=Adam(lr=0.0001)
adamax=Adamax(lr=0.0001)
nadam = Nadam(lr=0.0001)
classifier.compile(optimizer = adam, loss = 'binary_crossentropy', metrics = ['accuracy'])
#optimizer = algorítmo para selecionar o melhor peso do ANN. 'Adam' é um dos mais utilizados
#loss =  Algorítmo que minimiza as perdas do gradiente descendete estocástico. Como a saida é binária, utilizou binary_crossentropy. Se houver mais que uma variável categorical_crossentropy
#metrics = Padrão

#Fit classifier to the training test
history = classifier.fit(X_train, y_train, batch_size = 10, epochs = 100, validation_data=(X_test, y_test))

#batch_size = não tem um valor certo. 
#epochs  = não tem um valor certo

#Predicting the test set result
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5) #converte em verdadeiro ou falso

from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, precision_score, recall_score
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

results_final = pd.DataFrame([['ANN W/ Dropout', acc, prec, rec, f1]],
                      columns = ['Model', 'Accuracy', 'Precision', 'Recall', 'F1 Score'])

# Plot training & validation accuracy values
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.savefig('accuracy.png')
plt.show()

# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.savefig('loss.png')
plt.show()            
###########################################################################################################################                    
# Evaluating the ANN
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from keras.models import Sequential
from keras.layers import Dense
def build_classifier():
    classifier = Sequential()
    classifier.add(Dense(units = int((X_test.shape[1]/2)+1), kernel_initializer = 'uniform', activation = 'relu', 
                         input_dim = int(X_test.shape[1])))
    classifier.add(Dropout(p= 0.5))
    classifier.add(Dense(units = 24, kernel_initializer = 'uniform'))
    classifier.add(LeakyReLU(alpha=leaky_relu_alpha))
    classifier.add(Dropout(p= 0.5))
    classifier.add(Dense(units = 48, kernel_initializer = 'uniform'))
    classifier.add(LeakyReLU(alpha=leaky_relu_alpha))
    classifier.add(Dropout(p= 0.5))
    classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
    classifier.compile(optimizer = adam, loss = 'binary_crossentropy', metrics = ['accuracy'])
    return classifier
classifier = KerasClassifier(build_fn = build_classifier, batch_size = 10, epochs = 100)
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10, n_jobs = 1)
accuracies.mean()
accuracies.std()
accuracies_data = pd.Series(accuracies,name='Acurácia (%)')
accuracies_data = np.round(accuracies_data*100,0) 
accuracies_data.to_excel('Cross_validation.xlsx')   
print("ANN Accuracy: %0.3f (+/- %0.3f)" % (accuracies.mean(), accuracies.std() * 2))

import pydotplus
from keras.utils.vis_utils import model_to_dot
keras.utils.vis_utils.pydot = pydot
plot_model(classifier, to_file='model.png', show_shapes=True, show_layer_names=True)

#### Model Building ####
### Comparing Models

## Logistic Regression
from sklearn.linear_model import LogisticRegression
lr_classifier = LogisticRegression(random_state = 0, penalty = 'l2')
lr_classifier.fit(X_train, y_train)

# Predicting Test Set
y_pred = lr_classifier.predict(X_test)
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, precision_score, recall_score
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

model_results = pd.DataFrame([['Logistic Regression (Lasso)', acc, prec, rec, f1]],
               columns = ['Model', 'Accuracy', 'Precision', 'Recall', 'F1 Score'])

results_final = results_final.append(model_results, ignore_index = True)      

## K-Nearest Neighbors (K-NN)
#Choosing the K value
error_rate= []
for i in range(1,40):
    from sklearn.neighbors import KNeighborsClassifier
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train, y_train)
    pred_i = knn.predict(X_test)
    error_rate.append(np.mean(pred_i != y_test))
plt.figure(figsize=(10,6))
plt.plot(range(1,40),error_rate,color='blue', linestyle='dashed', marker='o',
         markerfacecolor='red', markersize=10)
plt.title('Error Rate vs. K Value')
plt.xlabel('K')
plt.ylabel('Error Rate')
print(np.mean(error_rate))

from sklearn.neighbors import KNeighborsClassifier
kn_classifier = KNeighborsClassifier(n_neighbors=15, metric='minkowski', p= 2)
kn_classifier.fit(X_train, y_train)

# Predicting Test Set
y_pred = kn_classifier.predict(X_test)
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

model_results = pd.DataFrame([['K-Nearest Neighbors (minkowski)', acc, prec, rec, f1]],
               columns = ['Model', 'Accuracy', 'Precision', 'Recall', 'F1 Score'])

results_final = results_final.append(model_results, ignore_index = True) 

## SVM (Linear)
from sklearn.svm import SVC
svm_linear_classifier = SVC(random_state = 0, kernel = 'linear', probability= True)
svm_linear_classifier.fit(X_train, y_train)

# Predicting Test Set
y_pred = svm_linear_classifier.predict(X_test)
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

model_results = pd.DataFrame([['SVM (Linear)', acc, prec, rec, f1]],
               columns = ['Model', 'Accuracy', 'Precision', 'Recall', 'F1 Score'])

results_final = results_final.append(model_results, ignore_index = True) 

## SVM (rbf)
from sklearn.svm import SVC
svm_rbf_classifier = SVC(random_state = 0, kernel = 'rbf', probability= True)
svm_rbf_classifier.fit(X_train, y_train)

# Predicting Test Set
y_pred = svm_rbf_classifier.predict(X_test)
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

model_results = pd.DataFrame([['SVM (RBF)', acc, prec, rec, f1]],
               columns = ['Model', 'Accuracy', 'Precision', 'Recall', 'F1 Score'])

results_final = results_final.append(model_results, ignore_index = True) 

## Naive Bayes
from sklearn.naive_bayes import GaussianNB
gb_classifier = GaussianNB()
gb_classifier.fit(X_train, y_train)

# Predicting Test Set
y_pred = gb_classifier.predict(X_test)
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

model_results = pd.DataFrame([['Naive Bayes (Gaussian)', acc, prec, rec, f1]],
               columns = ['Model', 'Accuracy', 'Precision', 'Recall', 'F1 Score'])

results_final = results_final.append(model_results, ignore_index = True) 

## Decision Tree
from sklearn.tree import DecisionTreeClassifier
dt_classifier = DecisionTreeClassifier(criterion='entropy', random_state=0)
dt_classifier.fit(X_train, y_train)

#Predicting the best set result
y_pred = dt_classifier.predict(X_test)
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

model_results = pd.DataFrame([['Decision Tree', acc, prec, rec, f1]],
               columns = ['Model', 'Accuracy', 'Precision', 'Recall', 'F1 Score'])

results_final = results_final.append(model_results, ignore_index = True) 


from sklearn.externals.six import StringIO  
from IPython.display import Image  
from sklearn.tree import export_graphviz
import pydotplus

dot_data = StringIO()
export_graphviz(dt_classifier, out_file=dot_data,  
                filled=True, rounded=True,
                special_characters=True)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
Image(graph.create_png())
graph.write_pdf('mestrado.pdf')
graph.write_png('mestrado.png')

## Random Forest
from sklearn.ensemble import RandomForestClassifier
rf_classifier = RandomForestClassifier(random_state = 0, n_estimators = 100,
                                    criterion = 'gini')
rf_classifier.fit(X_train, y_train)

# Predicting Test Set
y_pred = rf_classifier.predict(X_test)
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

model_results = pd.DataFrame([['Random Forest Gini (n=100)', acc, prec, rec, f1]],
               columns = ['Model', 'Accuracy', 'Precision', 'Recall', 'F1 Score'])

results_final = results_final.append(model_results, ignore_index = True)

## Ada Boosting
from sklearn.ensemble import AdaBoostClassifier
ad_classifier = AdaBoostClassifier()
ad_classifier.fit(X_train, y_train)

# Predicting Test Set
y_pred = ad_classifier.predict(X_test)
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

model_results = pd.DataFrame([['Ada Boosting', acc, prec, rec, f1]],
               columns = ['Model', 'Accuracy', 'Precision', 'Recall', 'F1 Score'])

results_final = results_final.append(model_results, ignore_index = True) 

##Gradient Boosting
from sklearn.ensemble import GradientBoostingClassifier
gr_classifier = GradientBoostingClassifier()
gr_classifier.fit(X_train, y_train)

# Predicting Test Set
y_pred = gr_classifier.predict(X_test)
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

model_results = pd.DataFrame([['Gradient Boosting', acc, prec, rec, f1]],
               columns = ['Model', 'Accuracy', 'Precision', 'Recall', 'F1 Score'])

results_final = results_final.append(model_results, ignore_index = True) 

##Ensemble Voting Classifier
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score
voting_classifier = VotingClassifier(estimators= [('lr', lr_classifier),
                                                  ('kn', kn_classifier),
                                                  ('svc_linear', svm_linear_classifier),
                                                  ('svc_rbf', svm_rbf_classifier),
                                                  ('gb', gb_classifier),
                                                  ('dt', dt_classifier),
                                                  ('rf', rf_classifier),
                                                  ('ad', ad_classifier),
                                                  ('gr', gr_classifier)], voting='soft')

for clf in (lr_classifier,kn_classifier,svm_linear_classifier,svm_rbf_classifier,
            gb_classifier, dt_classifier,rf_classifier, ad_classifier, gr_classifier, voting_classifier):
    clf.fit(X_train,y_train)
    y_pred = clf.predict(X_test)
    print(clf.__class__.__name__, accuracy_score(y_test, y_pred))

# Predicting Test Set
y_pred = voting_classifier.predict(X_test)
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

model_results = pd.DataFrame([['Ensemble Voting', acc, prec, rec, f1]],
               columns = ['Model', 'Accuracy', 'Precision', 'Recall', 'F1 Score'])

results_final = results_final.append(model_results, ignore_index = True)   

#The Best Classifier
print('The best classifier is:')
print('{}'.format(results_final.sort_values(by='Accuracy',ascending=False).head(5)))

#Applying K-fold validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator=rf_classifier, X=X_train, y=y_train,cv=10)
accuracies.mean()
accuracies.std()
print("Random Forest Gini (n=100) Accuracy: %0.3f (+/- %0.3f)" % (accuracies.mean(), accuracies.std() * 2))




















