# -*- coding: utf-8 -*-
"""
Created on Mon Nov 14 17:04:17 2022

@authors: arya, max, micah, yash
"""

import pandas as pd
import numpy as np
import scipy.stats
import scipy.stats as stats
from statsmodels.stats.power import tt_solve_power
from scipy.stats import anderson
from scipy.stats import levene
from scipy.stats import f_oneway
import pingouin as pg
from scipy.stats import chi2_contingency
import dowhy
from dowhy import CausalModel
from IPython.display import Image, display
import warnings

# PREPROCESSING THE DATA
dis = pd.read_csv('disney-characters.csv')
gross = pd.read_csv('disney_movies_total_gross.csv').dropna().reset_index()
for i in range(len(gross['inflation_adjusted_gross'])):
    gross['inflation_adjusted_gross'][i] = gross['inflation_adjusted_gross'][i].replace('$','')
    gross['inflation_adjusted_gross'][i] = gross['inflation_adjusted_gross'][i].replace(',','')
gross['inflation_adjusted_gross'] = pd.to_numeric(gross['inflation_adjusted_gross'])
print(gross['inflation_adjusted_gross'].describe())

# PART A: TWO-SAMPLE TWO-TAIL HYPOTHESIS
print('We have 513 movies accounted for in this dataset')
print('The mean inflation adjusted gross of Disney movies is roughly $127,594,100')
print('The worst performing Disney movie grossed $2,984')
print('The median Disney movie gross was  $59,679,130')
print('The highest grossing Disney movie made roughly $5,228,953,000')
print()
print('Is there a difference between the inflation adjusted gross of Disney movies from 1989-1999 and movies from 2000 onward')
print('H_0: There is no significant difference in gross')
print('H_a: There is difference in  gross revenue')

gross['release_date'] = pd.to_numeric(gross['release_date'].str[-4:])
era = gross[gross['release_date'] < 2000]
era = era[era['release_date'] >= 1989]
post = gross[gross['release_date'] >= 2000]

eraSamp = era['inflation_adjusted_gross'].sample(200)
postSamp = post['inflation_adjusted_gross'].sample(200)

# stat,p = stats.levene(era['inflation_adjusted_gross'],post['inflation_adjusted_gross'])
stat,p = stats.levene(eraSamp,postSamp)
print(p) # small pVal, not equal variances
# tStat2Tail,pVal2Tail = stats.ttest_ind(era['inflation_adjusted_gross'],post['inflation_adjusted_gross'],equal_var=False)
tStat2Tail,pVal2Tail = stats.ttest_ind(eraSamp,postSamp,equal_var=False)
print(f'pVal = {pVal2Tail}')
print('We reject the H_0, there  is a significant difference in gross revenues')

hypVal = 50000000
eraEffectSize = (np.mean(era['inflation_adjusted_gross'])-hypVal)/np.std(era['inflation_adjusted_gross']) #how do this w/o hypVal #BAYCIK SAID YOU CAN JUST MAKE UP A HYPVAL
eraSampSize = tt_solve_power(eraEffectSize,alpha=.05,power=.8,alternative='two-sided')
print()
print(eraSampSize)
print(f'We should get a smaller sample size, current effect size of {eraEffectSize} is pretty large')
postEffectSize = (np.mean(post['inflation_adjusted_gross'])-hypVal)/np.std(post['inflation_adjusted_gross'])
postSampSize = tt_solve_power(postEffectSize,alpha=.05,power=.8,alternative='two-sided')
print(postSampSize)
print(f'We should get a smaller sample size, current effect size of {postEffectSize} is pretty large')
print('Collected plenty of data')

print(np.mean(era['inflation_adjusted_gross']),np.mean(post['inflation_adjusted_gross']))


# PART B: ANOVA & TUKEY
# PREPROCESSING
rating1 = gross.iloc[:, np.r_[4:5, -1]] #indexing out MPAA_rating row and inflation adjusted gross income


G = rating1[rating1['MPAA_rating'] == 'G']
PG = rating1[rating1['MPAA_rating'] == 'PG']
PG13 = rating1[rating1['MPAA_rating'] == 'PG-13']
R = rating1[rating1['MPAA_rating'] == 'R']

flatrevenue =  rating1.iloc[0:,-1].values.flatten()

flatGrev = G.iloc[:,1].values.flatten()
flatPGrev = PG.iloc[:,1].values.flatten()
flatPG13rev = PG13.iloc[:,1].values.flatten()
flatRrev = R.iloc[:,1].values.flatten()

# ASSUMPTION CHECKS
Andersontest = scipy.stats.anderson(flatrevenue, dist='norm')
print("\nAnderson-Darling test: ", Andersontest)

Levenetest = levene(flatGrev,flatPGrev,flatRrev,flatPG13rev)
g = [np.var(x, ddof=1) for x in [flatGrev,flatPGrev,flatRrev,flatPG13rev]]
print(g)
print("\nLevene Test: ", Levenetest)

# OTHER STATS
# MEAN REVENUES OF MOVIES OF EACH RATING
print("\nMean gross rev for G rating: ", G.iloc[:,1].mean())
print("Mean gross rev for PG rating: ",PG.iloc[:,1].mean())
print("Mean gross rev for PG-13 rating: ",PG13.iloc[:,1].mean())
print("Mean gross rev for R rating: ",R.iloc[:,1].mean())

# NUMBERS OF MOVIES IN EACH RATING
print("\nNumber of G rated movies: ", len(G))
print("\nNumber of PG rated movies: ", len(PG))
print("\nNumber of PG-13 rated movies: ", len(PG13))
print("\nNumber of R rated movies: ", len(R))

# ANOVA TEST
OneWayAnovaTestStats1, OneWayAnovaPVal1 = f_oneway(G.iloc[:,1], PG.iloc[:,1], PG13.iloc[:,1], R.iloc[:,1])

print("ANOVA Test Stats: ", OneWayAnovaTestStats1)
print("ANOVA Pval: ", OneWayAnovaPVal1)

SignLevel = .05
if OneWayAnovaPVal1 <= SignLevel:
    print("We reject H0. The MPAA rating has impact on the gross revenue\n")
else:
    print("We do not reject H0. We cannot conclude that the training program has impact on the employee performance\n")


# getting rid NR in original data
indexNR = rating1[(rating1['MPAA_rating'] == 'Not Rated')].index
rating1.drop(indexNR , inplace=True)


# TUKEY-KRAMER TEST
pd.set_option("display.max_rows", None)
pd.set_option("display.max_column", None)
x = pg.pairwise_tukey(data=rating1, dv='inflation_adjusted_gross', between='MPAA_rating')
print(x)

# PART C: HETEROGENEITY
DisneyData = pd.read_csv("disney_movies_total_gross.csv",delimiter="\t")

gross = pd.read_csv('disney_movies_total_gross.csv')

gross = gross[gross['genre'].notna()]
gross = gross[gross['MPAA_rating'].notna()]

gross['inflation_adjusted_gross'] = gross['inflation_adjusted_gross'].map(lambda x: x.lstrip('$').rstrip('aAbBcC'))
gross = gross.replace(',','', regex=True)

gross['release_date'] = pd.to_datetime(gross['release_date'])
gross['release_date'] = gross['release_date'].dt.year
gross.rename(columns={'release_date': 'Year'}, inplace=True)

gross['inflation_adjusted_gross'] = gross['inflation_adjusted_gross'].astype(float)

gross['inflation_adjusted_gross'].describe()

MPAA_rating = gross.iloc[:,4]
inflation_adjusted_gross = gross.iloc[:,6]
index = gross.iloc[:,0]

Gover100 = []
Gunder100 = []
PGover100 = []
PGunder100 = []
PG13over100 = []
PG13under100 = []
Rover100 = []
Runder100 = []

for a in index:
    if MPAA_rating[a] == "G" and gross['inflation_adjusted_gross'][a] > 100000000:
        Gover100.append(a)
    
    if MPAA_rating[a] == "G" and gross['inflation_adjusted_gross'][a] < 100000000:
        Gunder100.append(a)
        
    if MPAA_rating[a] == "PG" and gross['inflation_adjusted_gross'][a] > 100000000:
        PGover100.append(a) 
    
    if MPAA_rating[a] == "PG" and gross['inflation_adjusted_gross'][a] < 100000000:
        PGunder100.append(a)
        
    if MPAA_rating[a] == "PG-13" and gross['inflation_adjusted_gross'][a] > 100000000:
        PG13over100.append(a)
        
    if MPAA_rating[a] == "PG-13" and gross['inflation_adjusted_gross'][a] < 100000000:
        PG13under100.append(a) 
        
    if MPAA_rating[a] == "R" and gross['inflation_adjusted_gross'][a] > 100000000:
        Rover100.append(a)
            
    if MPAA_rating[a] == "R" and gross['inflation_adjusted_gross'][a] < 100000000:
        Runder100.append(a) 

df = [[len(Gover100),len(Gunder100)],[len(PGover100), len(PGunder100)], 
      [len(PG13under100), len(PG13over100)],[len(Runder100), len(Rover100)]]

print (df)

teststat, pval, dof, expected_counts = chi2_contingency(df)

alpha = 0.1
print("the p value is ", pval)
if pval <= alpha:
    print('reject H0')
else:
    print('We fail to reject H0. The variables are heterogenous')

# PART D: ATE & REGRESSION
import statsmodels.api as sm
df = pd.read_excel('DisneySupplementalDataBaycik.xlsx')

#making preference a binary variable
preference = []
for x in df['Preference']:
    if x == 'Heroes':
        preference.append(1)
    else:
        preference.append(0)

df['Bi_Pref'] = preference

#dropping date because it is not helpful
df = df.drop('Date', axis = 1)
print(df)

print(df.isnull().sum())
df = df.dropna()
print(df.isnull().sum())

#changing variable into an int
df['Kids_younger_than_7'] = df['Kids_younger_than_7'].astype(int)

#LINEAR REGRESSION START
X = df.iloc[:, 1]
y = df.iloc[:, 4]

X = sm.add_constant(X)
#add intercept
est = sm.OLS(y, X)
#sm.OLS input is output y then input variable X
est2 = est.fit()
#est2 fits the regression model
print(est2.summary())

X1 = df.iloc[:, np.r_[1:2, 2:3]]
X1 = sm.add_constant(X1)
est3 = sm.OLS(y, X1)
est4 = est3.fit()
print(est4.summary())

causal_graph = """
digraph {
Kids_younger_than_7;
Retired;
Kids_younger_than_7 -> Bi_Pref;
Retired -> Bi_Pref
}
"""

model1 = CausalModel(
        data = df,
        graph=causal_graph.replace("\n", " "),
        treatment='Retired',
        outcome='Bi_Pref')
model1.view_model()
display(Image(filename="causal_model.png"))

# PART 3: DOWHY

warnings.filterwarnings('ignore')

causal_graph = """
digraph {
Kids_younger_than_7;
Age;
Retired;
Age -> Retired; Age -> Kids_younger_than_7;
Age -> Bi_Pref;
Kids_younger_than_7 -> Bi_Pref;
Retired -> Bi_Pref
}
"""

#Build the model in DoWhy
model= CausalModel(
        data = df,
        graph=causal_graph.replace("\n", " "),
        treatment='Kids_younger_than_7',
        outcome='Bi_Pref')
model.view_model()
display(Image(filename="causal_model.png"))#To create a png file of the causal model in the same folder as where the .py is 

#Identify the causal effect. Do calculus calculations in the background
estimands = model.identify_effect()
print(estimands)

#Causal Effect Estimation. There are few methods to Estimate the Causal Effect
causal_estimate_reg = model.estimate_effect(estimands,
                                            method_name="backdoor.linear_regression",
                                            test_significance=True)
print("Causal Estimate is " + str(causal_estimate_reg.value))

#Validation of assumptions. Add a placebo treatment. Replace the true treatment variable with an independent random variable; 
#If the assumption was correct, the estimate should go close to zero.

refute_results_reg = model.refute_estimate(estimands, causal_estimate_reg,
                                            method_name="placebo_treatment_refuter", num_simulations=100)
print(refute_results_reg)

#Validation of assumptions. Data Subset Refuter — replace the given dataset with a randomly selected subset; 
#If the assumption was correct, the estimation should not change that much.
refutel = model.refute_estimate(estimands,causal_estimate_reg, 
                                            method_name = "data_subset_refuter", num_simulations=100)
print(refutel)