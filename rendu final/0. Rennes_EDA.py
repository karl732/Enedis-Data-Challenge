
######################################################
##                    PROJET ENEDIS                 ##
##                                                  ##
##                      TEAM ESAK                   ##
######################################################


############################################
## 0) Importation de la base et nettoyage ## 
############################################

# In[Importation de base]

import pandas as pd

path = "/Users/leobriand/Documents/PROJETS/RENNES/1. RENNES_presentation/enedis"
path2 = "/Users/leobriand/Documents/PROJETS/RENNES/4. RENNES_diapo"

data = pd.read_excel(path+"/data_train.xlsx")
test = pd.read_excel(path+"/data_test.xlsx")

data.shape

# In[Traitements préliminaires : Renommage, Valeurs manquantes, Doublons]


# RENOMMAGE


# Nous renommons nos colonnes à l'aide du descriptif court disponible en ressources

renom = {
    'ID_t': 'Identifiant du tronçon',
    'Nb_of_incident': 'Nombre d’incident sur le départ',
    'Service_date': 'Date de mise en exploitation du tronçon',
    'Length_fragile_section': 'Longueur section fragile sur le départ',
    'Year_helicopter_flight': 'Année du dernier vol sur le départ',
    'Electrical_length': 'Longueur électrique',
    'Length_climate_hazard_plan': 'Longueur en plan aléa climatique sur le départ',
    'Nb_of_anomaly': "Nombre d'anomalies sur le départ",
    'Last treatment PR immobilized': 'Année de la dernière maintenance sur le départ',
}

data = data.rename(columns=renom)
del renom, path


# VALEURS MANQUANTES

# Nous calculons la proportion de valeurs manquantes dans pour chaque variable

nan = (data.isna().mean() * 100).round(1)
for column, percentage in nan.items():
    print(f"{column}: {percentage}%")
    
    # Nous disposons de 50% d'observations manquantes pour notre variable
    # "Nombre_d_incidents", 68% pour notre variable "Nombre_d_anomalies"
    # et 89% pour la dernière date de traitement
    
data['Nombre d’incident sur le départ'] = data['Nombre d’incident sur le départ'].fillna(0)
data["Nombre d'anomalies sur le départ"] = data["Nombre d'anomalies sur le départ"].fillna(0)
data['Année de la dernière maintenance sur le départ'] = data['Année de la dernière maintenance sur le départ'].fillna(0)


# DOUBLONS


# Nous nous assurons de ne pas observer de doublons dans la base de données brute
doublons = data.duplicated().any()
    
if doublons:
    nb_doublons = data.duplicated().sum()
    aff_doublons = data[data.duplicated(keep=False)]
    
    # Nous affichons le nombre de doublons: print(aff_doublons)
    print(f"Il y a {nb_doublons} doublons.")
    
    data.drop_duplicates(inplace=True)
    # Nous affichons les observations dupliquées : print(aff_doublons)

else:
    print("Il n'y a pas de doublons dans la table.")

# Nous vérifions que les doublons, si il y en a, ont bien été supprimés
doublons = data.duplicated().any()

if doublons:
    nb_doublons = data.duplicated().sum()
    print(f"Il y a {nb_doublons} doublons.")

else:
    print("Les doublons ont été supprimés de la base.")
    print("\n")


# Étudions les duplicats pour la variable "Identifiant du tronçon"
moda_id = set(data["Identifiant du tronçon"])

if len(moda_id) == len(data["Identifiant du tronçon"]):
    print("Il n'y a aucune modalité qui se répète pour la variable 'Identifiant du tronçon'. Nous pouvons la supprimer.")

else:
    doublons_id = data[data.duplicated(subset=["Identifiant du tronçon"], keep=False)]
    nb_doublons_id = data.duplicated(subset=["Identifiant du tronçon"], keep=False).sum()
    print(f"{nb_doublons_id} modalité(s) se répète(nt) pour la variable 'Identifiant du tronçon'.")
    
    data = data.drop_duplicates(subset=["Identifiant du tronçon"], keep='first')
    
# Nous vérifions que si il y a des doublons pour la variable "Identifiant du tronçon", ces derniers ont bien été supprimés
moda_id = set(data["Identifiant du tronçon"])

if len(moda_id) == len(data):
    print("Il n'y a plus aucune modalité qui se répète pour la variable 'Identifiant du tronçon'.")
    
else:
    print("Les doublons ont été supprimés de la base.")
    print("\n")
    
data_doublons = data.copy()





# In[Statistiques descriptives variable cible]

data_inc = data.groupby(data['Nombre d’incident sur le départ']).size().reset_index(name='Nombre')
data_inc['Proportion'] = (data_inc['Nombre'] / data_inc['Nombre'].sum()) * 100

# Au minimum, un tronçon a subit 1 anomalie
# Au plus, il en a subit 6, c'est le cas pour 0.3% des tronçons de la base

import seaborn as sns
from matplotlib import pyplot as plt

plot = plt.pie(data=data_inc, x="Proportion", labels="Nombre d’incident sur le départ",autopct='%1.1f%%', startangle=140, colors=sns.color_palette('viridis'))
plt.savefig(path2 + 'camembert_incident.pdf')
plt.title("Occurences des incidents par tronçon")
plt.show()




# In[Statistiques descriptives pour les prédicteurs]


# Exploitation


data['Date de mise en exploitation du tronçon'] = pd.to_datetime(data['Date de mise en exploitation du tronçon'], errors='coerce')
data['Année de mise en exploitation du tronçon'] = data['Date de mise en exploitation du tronçon'].dt.year

data_date_expl = data.groupby(data['Année de mise en exploitation du tronçon']).size().reset_index(name='Nombre')

import seaborn as sns
from matplotlib import pyplot as plt


plot = sns.lineplot(data=data_date_expl, x="Année de mise en exploitation du tronçon", y="Nombre", palette="viridis")
plot.spines[['top', 'right', 'bottom', 'left']].set_visible(False)
plot.xaxis.grid(alpha=0.2)
plot.set_xlabel(None)
plot.yaxis.grid(alpha=0.2)
plot.set_ylabel("Nombre de lignes mises en service")
plt.title("Tronçons mis en service par année")
plt.savefig(path2 + 'lineplot_expltot.pdf')
plt.show()



data_date_incidents = data.groupby(['Année de mise en exploitation du tronçon', 'Nombre d’incident sur le départ']).size().reset_index(name='Nombre')

plot = sns.lineplot(x='Année de mise en exploitation du tronçon', y='Nombre', hue='Nombre d’incident sur le départ', data=data_date_incidents, palette='viridis_r', ci=None)
plot.spines[['top', 'right', 'bottom', 'left']].set_visible(False)
plot.xaxis.grid(alpha=0.2)
plot.set_xlabel(None)
plot.yaxis.grid(alpha=0.2)
plot.set_ylabel("Tronçons mis en exploitation")
plt.savefig(path2 + 'lineplot_expl.pdf')
plt.title('Nombre de tronçons mis en exploitation par année')
plt.legend(title="Nombre d'incidents", loc='upper right', bbox_to_anchor=(1.2, 1))
plt.savefig(path2 + 'lineplot_expltot.pdf')
plt.show()





# In[]

# Maintenance : BARPLOT 

data_date_mtn = data.groupby(data['Année de la dernière maintenance sur le départ']).size().reset_index(name='Nombre')
data_date_mtn = data_date_mtn.iloc[1:]

plot = sns.barplot(data=data_date_mtn, x="Année de la dernière maintenance sur le départ", y="Nombre", palette="viridis")
plot.spines[['top', 'right', 'bottom', 'left']].set_visible(False)
plot.set_xlabel(None)
plot.yaxis.grid(alpha=0.2)
plot.set_ylabel("Dernières maintenances réalisées")
plt.savefig(path2 + 'barplot_mtn.pdf')
plt.title("Répartition des dernières maintenances réalisées par année")
plt.show()



# Hélico : BARPLOT 

data_date_helico = data.groupby(data['Année du dernier vol sur le départ']).size().reset_index(name='Nombre')

plot = sns.barplot(data=data_date_helico, x="Année du dernier vol sur le départ", y="Nombre", palette="viridis")
plot.spines[['top', 'right', 'bottom', 'left']].set_visible(False)
plot.set_xlabel(None)
plot.yaxis.grid(alpha=0.2)
plot.set_ylabel("Derniers vols d'hélicoptère initiés")
plt.title("Répartition des derniers vols d'hélicoptère initiés par année")
plt.savefig(path2 + 'barplot_helico.pdf')
plt.show()


data_date_helico = data.groupby(['Année du dernier vol sur le départ', 'Nombre d’incident sur le départ']).size().reset_index(name='Nombre')

plot = sns.barplot(data=data_date_helico, x="Année du dernier vol sur le départ", y="Nombre", hue="Nombre d’incident sur le départ", palette="viridis")
plot.spines[['top', 'right', 'bottom', 'left']].set_visible(False)
plot.set_xlabel(None)
plot.yaxis.grid(alpha=0.2)
plot.set_ylabel("Derniers vol d'hélicoptère initiés")
plt.title("Répartition des vols d'hélicoptère initiés par année")
plt.savefig(path2 + 'barplot_helico2.pdf')
plt.show()


# DATE Hélico

data_date_helico = data.groupby(['Année du dernier vol sur le départ', 'Nombre d’incident sur le départ']).size().reset_index(name='Nombre')

plot = sns.lineplot(data=data_date_helico, x="Année du dernier vol sur le départ", y="Nombre", hue="Nombre d’incident sur le départ", palette="viridis")
plot.spines[['top', 'right', 'bottom', 'left']].set_visible(False)
plot.set_xlabel(None)
plot.yaxis.grid(alpha=0.2)
plt.title("Répartition des derniers vols d'hélicoptère initiés par année")
plot.set_ylabel("Derniers vols d'hélicoptères initiés")
plt.savefig(path2 + 'lineplot_helico.pdf')
plt.show()



# In[]

plot = sns.boxplot(x='Nombre d’incident sur le départ', y='Longueur section fragile sur le départ', data=data)
plt.savefig(path2 + 'boxplot_fragile.pdf')
plt.title('Boxplot : la longueur des sections fragiles')

plot = sns.boxplot(x='Nombre d’incident sur le départ', y='Longueur électrique', data=data)
plt.savefig(path2 + 'boxplot_elec.pdf')
plt.title('Boxplot : la longueur électrique')

plot = sns.boxplot(x='Nombre d’incident sur le départ', y='Longueur en plan aléa climatique sur le départ', data=data)
plt.savefig(path2 + 'boxplot_climat.pdf')
plt.title('Boxplot : la longueur du plan aléa climatique')


# In[]

# SCATTER-PLOT


plot = sns.scatterplot(data=data, x="Longueur section fragile sur le départ", y="Longueur en plan aléa climatique sur le départ", hue="Nombre d’incident sur le départ", palette="viridis")
plot.spines[['top', 'right', 'bottom', 'left']].set_visible(False)
plot.xaxis.grid(alpha=0.2)
plot.set_xlabel(None)
plot.yaxis.grid(alpha=0.2)
plot.set_ylabel("Nombre d’incidents sur le départ")
plt.title("Nombre d’incidents / Nombre d'anomalies")
plt.savefig(path2 + 'scatter_.pdf')
plt.show()




# In[Regroupement de modalités pour la variable cible]
# Après avis de ENEDIS, nous passons la variable 'Nombre d’incident sur le départ' en binaire.

def binaire(valeur):
    return 0 if valeur == 0 else 1

data['Nombre d’incident sur le départ'] = data['Nombre d’incident sur le départ'].apply(binaire)


# In[Statistiques descriptives variable cible]

data_inc = data.groupby(data['Nombre d’incident sur le départ']).size().reset_index(name='Nombre')
data_inc['Proportion'] = (data_inc['Nombre'] / data_inc['Nombre'].sum()) * 100


import seaborn as sns
from matplotlib import pyplot as plt

plot = plt.pie(data=data_inc, x="Proportion", labels="Nombre d’incident sur le départ",autopct='%1.1f%%', startangle=140, colors=sns.color_palette('viridis'))
plt.savefig(path2 + 'camembert_incident.pdf')
plt.title("Occurences des incidents par tronçon")
plt.show()

# In[Statistiques descriptives pour la variable cible "Identifiant du tronçon"]


# BOXPLOT de la cible avec les variables 'Longueur"


plot = sns.boxplot(x='Nombre d’incident sur le départ', y='Longueur section fragile sur le départ', data=data)
plt.savefig(path2 + 'boxplot_fragilebi.pdf')
plt.title('Boxplot : la longueur des sections fragiles')

plot = sns.boxplot(x='Nombre d’incident sur le départ', y='Longueur électrique', data=data)
plt.savefig(path2 + 'boxplot_elecbi.pdf')
plt.title('Boxplot : la longueur électrique')

plot = sns.boxplot(x='Nombre d’incident sur le départ', y='Longueur en plan aléa climatique sur le départ', data=data)
plt.savefig(path2 + 'boxplot_climatbi.pdf')
plt.title('Boxplot : la longueur du plan aléa climatique')



# In[Statistiques descriptives pour les prédicteurs et la cible binomiale]


# Mise en exploitation : LINEPLOT

data['Date de mise en exploitation du tronçon'] = pd.to_datetime(data['Date de mise en exploitation du tronçon'], errors='coerce')
data['Année de mise en exploitation du tronçon'] = data['Date de mise en exploitation du tronçon'].dt.year

data_date_incidents = data.groupby(['Année de mise en exploitation du tronçon', 'Nombre d’incident sur le départ']).size().reset_index(name='Nombre')

plot = sns.lineplot(x='Année de mise en exploitation du tronçon', y='Nombre', hue='Nombre d’incident sur le départ', data=data_date_incidents, palette='viridis_r', ci=None)
plot.spines[['top', 'right', 'bottom', 'left']].set_visible(False)
plot.xaxis.grid(alpha=0.2)
plot.set_xlabel(None)
plot.yaxis.grid(alpha=0.2)
plot.set_ylabel("Nombre de lignes mises en service")
plt.title("Tronçons mis en service par année")
plt.legend(title="Nombre d'incidents", loc='upper right', bbox_to_anchor=(1.2, 1))
plt.savefig(path2 + 'barplot_explbi.pdf')
plt.show()



# Maintenance : BARPLOT / LINEPLOT 

data_date_mtn = data.groupby(['Année de la dernière maintenance sur le départ', 'Nombre d’incident sur le départ']).size().reset_index(name='Nombre')
data_date_mtn = data_date_mtn.iloc[2:]

plot = sns.barplot(data=data_date_mtn, x="Année de la dernière maintenance sur le départ", y="Nombre", hue="Nombre d’incident sur le départ", palette="viridis")
plot.spines[['top', 'right', 'bottom', 'left']].set_visible(False)
plot.set_xlabel(None)
plot.yaxis.grid(alpha=0.2)
plot.set_ylabel("Nombre d’incidents sur le départ")
plt.title("Nombre d’incidents l'année de la dernière maintenance sur le départ")
plt.savefig(path2 + 'barplot_mtn.pdf')
plt.show()


plot = sns.lineplot(data=data_date_mtn, x="Année de la dernière maintenance sur le départ", y="Nombre", hue="Nombre d’incident sur le départ", palette="viridis")
plot.spines[['top', 'right', 'bottom', 'left']].set_visible(False)
plot.set_xlabel(None)
plot.yaxis.grid(alpha=0.2)
plot.set_ylabel("Nombre d’incidents sur le départ")
plt.title("Nombre d’incidents l'année de la dernière maintenance sur le départ")
plt.savefig(path2 + 'lineplot_mtnbi.pdf')
plt.show()


# Hélicoptère : BARPLOT 

data_date_helico = data.groupby(['Année du dernier vol sur le départ', 'Nombre d’incident sur le départ']).size().reset_index(name='Nombre')
data_date_helico = data_date_helico.iloc[2:]

plot = sns.barplot(data=data_date_helico, x="Année du dernier vol sur le départ", y="Nombre", hue="Nombre d’incident sur le départ", palette="viridis")
plot.spines[['top', 'right', 'bottom', 'left']].set_visible(False)
plot.set_xlabel(None)
plot.yaxis.grid(alpha=0.2)
plot.set_ylabel("Nombre d’incidents sur le départ")
plt.title("Nombre d’incidents l'année du dernier vol sur le départ")
plt.savefig(path2 + 'barplot_helicobi.pdf')
plt.show()


data_date_mtn.to_excel(path2 + '/data_graph.xlsx', index=True)

