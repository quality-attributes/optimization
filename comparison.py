# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.3.2
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %%
import numpy as np
import pandas as pd
from bokeh.plotting import figure, show
from bokeh.io import output_notebook
from bokeh.models.tools import HoverTool

resultados = list()
ejecuciones = list()

for i in range(30):
    resultados.append(pd.read_csv('./executions/ed-rand-1bin/exec_' + str(i + 1) + '.csv'))
    resultados[i]['execution'] = (i + 1)
    ejecuciones.append(resultados[i].tail(100)[['execution','alpha', 'fit_prior', 'fitness']])
    
output_notebook()

# %%
df = pd.DataFrame
df = pd.concat(ejecuciones, ignore_index=True)

# %%
promedio = []
mejor = []
peor = []
desviacion = []
mediana = []

for i in range(30):
    mejor.append(df.loc[df['execution'] == i + 1].max().values[3])
    promedio.append(df.loc[df['execution'] == i + 1].mean().values[3])
    desviacion.append(df.loc[df['execution'] == i + 1].std().values[3])
    mediana.append(df.loc[df['execution'] == i + 1].median().values[3])
    peor.append(df.loc[df['execution'] == i + 1].min().values[3])

data = {'Número de Ejecución': list(range(1, 31)),\
    'Mejor':mejor, \
    'Promedio':promedio, \
    'Mediana':mediana, \
    'Desviación Estándar':desviacion, \
    'Peor':peor}
df = pd.DataFrame(data)

df.set_index(['Número de Ejecución'], inplace=True)
df = df.sort_values("Mejor", ascending = True)
df

# %%
df.describe()

# %%
df_mejor = df.loc[df["Mejor"] == df["Mejor"].min()].head(1)
df_mejor

# %%
mejor_ejecucion = df_mejor.index.values[0]
df2 = pd.read_csv('./executions/ed-rand-1bin/exec_6.csv')

promedio = []
mejor = []
peor = []
desviacion = []
mediana = []
alpha = []
fit_prior = []
gen = []


for i in range(len(df2['generation'].unique().tolist())):
    alpha.append(df2.loc[df2['generation'] == i + 1].head(1).alpha.values[0])
    fit_prior.append(df2.loc[df2['generation'] == i + 1].head(1).fit_prior.values[0])
    mejor.append(df2.loc[df2['generation'] == i + 1].min().values[3])
    promedio.append(df2.loc[df2['generation'] == i + 1].mean().values[3])
    desviacion.append(df2.loc[df2['generation'] == i + 1].std().values[3])
    mediana.append(df2.loc[df2['generation'] == i + 1].median().values[3])
    peor.append(df2.loc[df2['generation'] == i + 1].max().values[3])
    gen.append(i+1)

data = {'Generation': gen,\
    'Mejor':mejor, \
    'Promedio':promedio, \
    'Mediana':mediana, \
    'Desviación Estándar':desviacion, \
    'Peor':peor, \
    'alpha': alpha, \
    'fit_prior': fit_prior}
df2 = pd.DataFrame(data)
df2.head()

# %%

p = figure(title = "Desempeño de Mejor Ejecución", plot_width=800, plot_height=500)
p.xaxis.axis_label = 'Función Aptitud'
p.yaxis.axis_label = 'Generation'

p.line('Generation','Mejor', source=df2, line_width=2)

hover=HoverTool()
hover.tooltips=[
    ('Generación','@Generation'),
    ('alpha','@alpha'),
    ('fir_prior', '@fit_prior'),
    ('Aptitud','@Mejor'),
]
p.add_tools(hover)
show(p)

# %%
df_peor = df.loc[df["Mejor"] == df["Mejor"].max()].head(1)
df_peor

# %%
peor_ejecucion = df_peor.index.values[0]
df3 = pd.read_csv('./executions/ed-rand-1bin/exec_30.csv')

promedio = []
mejor = []
peor = []
desviacion = []
mediana = []
alpha = []
fit_prior = []
gen = []


for i in range(len(df3['generation'].unique().tolist())):
    alpha.append(df3.loc[df3['generation'] == i + 1].head(1).alpha.values[0])
    fit_prior.append(df3.loc[df3['generation'] == i + 1].head(1).fit_prior.values[0])
    mejor.append(df3.loc[df3['generation'] == i + 1].min().values[3])
    promedio.append(df3.loc[df3['generation'] == i + 1].mean().values[3])
    desviacion.append(df3.loc[df3['generation'] == i + 1].std().values[3])
    mediana.append(df3.loc[df3['generation'] == i + 1].median().values[3])
    peor.append(df3.loc[df3['generation'] == i + 1].max().values[3])
    gen.append(i+1)

data = {'Generation': gen,\
    'Mejor':mejor, \
    'Promedio':promedio, \
    'Mediana':mediana, \
    'Desviación Estándar':desviacion, \
    'Peor':peor, \
    'alpha': alpha, \
    'fit_prior': fit_prior}
df3 = pd.DataFrame(data)
df3.head()

# %%
p = figure(title = "Desempeño de Peor Ejecución", plot_width=800, plot_height=500)
p.xaxis.axis_label = 'Función Aptitud'
p.yaxis.axis_label = 'Generation'

p.line('Generation','Mejor', source=df3, line_width=2)

hover=HoverTool()
hover.tooltips=[
    ('Generación','@Generation'),
    ('alpha','@alpha'),
    ('fir_prior', '@fit_prior'),
    ('Aptitud','@Mejor'),
]
p.add_tools(hover)
show(p)

# %%
df4 = pd.read_csv('./executions/ed-rand-1bin/exec_16.csv')

promedio = []
mejor = []
peor = []
desviacion = []
mediana = []
alpha = []
fit_prior = []
gen = []


for i in range(len(df4['generation'].unique().tolist())):
    alpha.append(df4.loc[df4['generation'] == i + 1].head(1).alpha.values[0])
    fit_prior.append(df4.loc[df4['generation'] == i + 1].head(1).fit_prior.values[0])
    mejor.append(df4.loc[df4['generation'] == i + 1].min().values[3])
    promedio.append(df4.loc[df4['generation'] == i + 1].mean().values[3])
    desviacion.append(df4.loc[df4['generation'] == i + 1].std().values[3])
    mediana.append(df4.loc[df4['generation'] == i + 1].median().values[3])
    peor.append(df4.loc[df4['generation'] == i + 1].max().values[3])
    gen.append(i+1)

data = {'Generation': gen,\
    'Mejor':mejor, \
    'Promedio':promedio, \
    'Mediana':mediana, \
    'Desviación Estándar':desviacion, \
    'Peor':peor, \
    'alpha': alpha, \
    'fit_prior': fit_prior}
df4 = pd.DataFrame(data)
df4.head()

# %%
p = figure(title = "Desempeño de Peor Ejecución", plot_width=800, plot_height=500)
p.xaxis.axis_label = 'Función Aptitud'
p.yaxis.axis_label = 'Generation'

p.line('Generation','Mejor', source=df4, line_width=2)

hover=HoverTool()
hover.tooltips=[
    ('Generación','@Generation'),
    ('alpha','@alpha'),
    ('fir_prior', '@fit_prior'),
    ('Aptitud','@Mejor'),
]
p.add_tools(hover)
show(p)

# %%
p = figure(title = "Gráfica de Convergencia", plot_width=800, plot_height=500)

font_size="15px"
p.title.text_font_size = "18px"
p.xaxis.axis_label = 'Date'
p.xaxis.axis_label_text_font_size = font_size
p.xaxis.major_label_text_font_size = font_size
p.yaxis.axis_label = 'Position'
p.yaxis.axis_label_text_font_size = font_size
p.yaxis.major_label_text_font_size = font_size
p.xaxis.axis_label = 'Generación'
p.yaxis.axis_label = 'CV Accuracy'

p.line('Generation','Mejor', source=df2, line_width=2.5, line_color='#DBB13B', legend_label='Mejor')
p.line('Generation','Mejor', source=df4, line_width=2.5, line_color='#5B6A9A', legend_label='Mediana')
p.line('Generation','Mejor', source=df3, line_width=2.5, line_color='#4A90E2', legend_label='Peor')

hover=HoverTool()
hover.tooltips=[
    ('Generación','@Generation'),
    ('alpha','@alpha'),
    ('fit_prior', '@fit_prior'),
    ('Aptitud','@Mejor'),
]

p.legend.location = "bottom_right"
p.legend.label_text_font_size = "17px"

p.add_tools(hover)
show(p)

# %% [markdown]
# ## TPOT

# %%
resultados = list()
ejecuciones = list()

for i in range(30):
    resultados.append(pd.read_csv('./executions/tpot/tpot_' + str(i + 1) + '.csv'))
    resultados[i]['execution'] = (i + 1)
    ejecuciones.append(resultados[i].tail(100)[['execution', 'fitness']])

# %%
df = pd.DataFrame
df = pd.concat(ejecuciones, ignore_index=True)

# %%
promedio = []
mejor = []
peor = []
desviacion = []
mediana = []

for i in range(30):
    mejor.append(df.loc[df['execution'] == i + 1].max().values[1])
    promedio.append(df.loc[df['execution'] == i + 1].mean().values[1])
    desviacion.append(df.loc[df['execution'] == i + 1].std().values[1])
    mediana.append(df.loc[df['execution'] == i + 1].median().values[1])
    peor.append(df.loc[df['execution'] == i + 1].min().values[1])

data = {'Número de Ejecución': list(range(1, 31)),\
    'Mejor':mejor, \
    'Promedio':promedio, \
    'Mediana':mediana, \
    'Desviación Estándar':desviacion, \
    'Peor':peor}
df = pd.DataFrame(data)

df.set_index(['Número de Ejecución'], inplace=True)
df = df.sort_values("Mejor", ascending = False)
df

# %%
df5 = pd.read_csv('./executions/tpot/tpot_9.csv')

promedio = []
mejor = []
peor = []
desviacion = []
mediana = []
gen = []


for i in range(len(df5['generation'].unique().tolist())):
    mejor.append(df5.loc[df5['generation'] == i + 1].min().values[1])
    promedio.append(df5.loc[df5['generation'] == i + 1].mean().values[1])
    desviacion.append(df5.loc[df5['generation'] == i + 1].std().values[1])
    mediana.append(df5.loc[df5['generation'] == i + 1].median().values[1])
    peor.append(df5.loc[df5['generation'] == i + 1].max().values[1])
    gen.append(i+1)

data = {'Generation': gen,\
    'Mejor':mejor, \
    'Promedio':promedio, \
    'Mediana':mediana, \
    'Desviación Estándar':desviacion, \
    'Peor':peor}
df5 = pd.DataFrame(data)
df5.head()

# %%
p = figure(title = "Desempeño de Mejor Ejecución", plot_width=800, plot_height=500)
p.xaxis.axis_label = 'Función Aptitud'
p.yaxis.axis_label = 'Generation'

p.line('Generation','Mejor', source=df5, line_width=2)

hover=HoverTool()
hover.tooltips=[
    ('Generación','@Generation'),
    ('Aptitud','@Mejor'),
]
p.add_tools(hover)
show(p)

# %%
df6 = pd.read_csv('./executions/tpot/tpot_4.csv')

promedio = []
mejor = []
peor = []
desviacion = []
mediana = []
gen = []


for i in range(len(df6['generation'].unique().tolist())):
    mejor.append(df6.loc[df6['generation'] == i + 1].min().values[1])
    promedio.append(df6.loc[df6['generation'] == i + 1].mean().values[1])
    desviacion.append(df6.loc[df6['generation'] == i + 1].std().values[1])
    mediana.append(df6.loc[df6['generation'] == i + 1].median().values[1])
    peor.append(df6.loc[df6['generation'] == i + 1].max().values[1])
    gen.append(i+1)

data = {'Generation': gen,\
    'Mejor':mejor, \
    'Promedio':promedio, \
    'Mediana':mediana, \
    'Desviación Estándar':desviacion, \
    'Peor':peor}
df6 = pd.DataFrame(data)
df6.head()

# %%
p = figure(title = "Desempeño de Peor Ejecución", plot_width=800, plot_height=500)
p.xaxis.axis_label = 'Función Aptitud'
p.yaxis.axis_label = 'Generation'

p.line('Generation','Mejor', source=df6, line_width=2)

hover=HoverTool()
hover.tooltips=[
    ('Generación','@Generation'),
    ('Aptitud','@Mejor'),
]
p.add_tools(hover)
show(p)

# %%
df7 = pd.read_csv('./executions/tpot/tpot_23.csv')

promedio = []
mejor = []
peor = []
desviacion = []
mediana = []
gen = []


for i in range(len(df7['generation'].unique().tolist())):
    mejor.append(df7.loc[df7['generation'] == i + 1].min().values[1])
    promedio.append(df7.loc[df7['generation'] == i + 1].mean().values[1])
    desviacion.append(df7.loc[df7['generation'] == i + 1].std().values[1])
    mediana.append(df7.loc[df7['generation'] == i + 1].median().values[1])
    peor.append(df7.loc[df7['generation'] == i + 1].max().values[1])
    gen.append(i+1)

data = {'Generation': gen,\
    'Mejor':mejor, \
    'Promedio':promedio, \
    'Mediana':mediana, \
    'Desviación Estándar':desviacion, \
    'Peor':peor}
df7 = pd.DataFrame(data)
df7.head()

# %%
p = figure(title = "Desempeño de Mediana Ejecución", plot_width=800, plot_height=500)
p.xaxis.axis_label = 'Función Aptitud'
p.yaxis.axis_label = 'Generation'

p.line('Generation','Mejor', source=df7, line_width=2)

hover=HoverTool()
hover.tooltips=[
    ('Generación','@Generation'),
    ('Aptitud','@Mejor'),
]
p.add_tools(hover)
show(p)

# %%
p = figure(title = "Gráfica de Convergencia", plot_width=800, plot_height=500)

font_size="15px"
p.title.text_font_size = "18px"
p.xaxis.axis_label = 'Date'
p.xaxis.axis_label_text_font_size = font_size
p.xaxis.major_label_text_font_size = font_size
p.yaxis.axis_label = 'Position'
p.yaxis.axis_label_text_font_size = font_size
p.yaxis.major_label_text_font_size = font_size
p.xaxis.axis_label = 'Generación'
p.yaxis.axis_label = 'CV Score'

p.line('Generation','Mejor', source=df5, line_width=2.5, line_color='#DBB13B', legend_label='Mejor')
p.line('Generation','Mejor', source=df7, line_width=2.5, line_color='#5B6A9A', legend_label='Mediana')
p.line('Generation','Mejor', source=df6, line_width=2.5, line_color='#4A90E2', legend_label='Peor')

hover=HoverTool()
hover.tooltips=[
    ('Generación','@Generation'),
    ('Aptitud','@Mejor'),
]

p.legend.location = "bottom_right"
p.legend.label_text_font_size = "17px"

p.add_tools(hover)
show(p)

# %%
