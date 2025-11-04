import numpy as np
import pandas as pd
df = pd.read_csv("MCDO/survey.csv")

columnas_relevantes = [
    'work_interfere', 'family_history', 'treatment', 'remote_work',
    'tech_company', 'benefits', 'care_options', 'wellness_program',
    'seek_help', 'leave', 'mental_health_consequence', 'phys_health_consequence',
    'obs_consequence', 'coworkers', 'supervisor',
    'mental_health_interview', 'phys_health_interview', 'mental_vs_physical'
]

burnout_mapping = {
    'work_interfere': {'Never': 0, 'Rarely': 0.33, 'Sometimes': 0.66, 'Often': 1},
    'family_history': {'No': 0, 'Yes': 1},
    'treatment': {'No': 0, 'Yes': 1},
    'remote_work': {'No': 0, 'Yes': 0.5},
    'tech_company': {'Yes': 0.5, 'No': 0},
    'benefits': {'Yes': 0, "Don't know": 0.5, 'No': 1},
    'care_options': {'Yes': 0, 'Not sure': 0.5, 'No': 1},
    'wellness_program': {'No': 1, "Don't know": 0.5, 'Yes': 0},
    'seek_help': {'Yes': 0, "Don't know": 0.5, 'No': 1},
    'leave': {'Very easy': 0, 'Somewhat easy': 0.33, "Don't know": 0.5,
              'Somewhat difficult': 0.66, 'Very difficult': 1},
    'mental_health_consequence': {'No': 0, 'Maybe': 0.5, 'Yes': 1},
    'phys_health_consequence': {'No': 0, 'Maybe': 0.5, 'Yes': 1},
    'obs_consequence': {'No': 0, 'Yes': 1},
    'coworkers': {'Yes': 0, 'Some of them': 0.5, 'No': 1},
    'supervisor': {'Yes': 0, 'Some of them': 0.5, 'No': 1},
    'mental_health_interview': {'No': 0, 'Maybe': 0.5, 'Yes': 1},
    'phys_health_interview': {'No': 0, 'Maybe': 0.5, 'Yes': 1},
    'mental_vs_physical': {'Yes': 0, "Don't know": 0.5, 'No': 1}
}

def clasificar_edad(edad):
    if pd.isna(edad):
        return np.nan
    elif edad < 25:
        return "18-24"
    elif edad < 35:
        return "25-34"
    elif edad < 45:
        return "35-44"
    elif edad < 55:
        return "45-54"
    else:
        return "55+"

def limpiar_genero(g):
    if not isinstance(g, str):
        return np.nan
    g = g.strip().lower()

    # Palabras clave asociadas a "hombre"
    male_terms = ["male", "m", "man", "mal", "maile", "male-ish", "cis male", 
                  "cis man", "guy", "msle", "make", "malr", "mail"]
    
    # Palabras clave asociadas a "mujer"
    female_terms = ["female", "f", "woman", "cis female", "cis woman", 
                    "femake", "femail", "female (trans)", "female ", "womyn"]
    
    # Si contiene palabras masculinas
    if any(term in g for term in male_terms):
        return "Male"
    # Si contiene palabras femeninas
    elif any(term in g for term in female_terms):
        return "Female"
    # Si parece no binario, queer o indefinido
    elif any(term in g for term in ["non", "gender", "fluid", "trans", "queer", "androgyne", "agender", "enby", "neuter", "other"]):
        return "Other"
    else:
        return "Other"
    
