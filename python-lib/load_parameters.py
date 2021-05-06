import dataiku
from dataiku.customrecipe import (
    get_recipe_config
)

def load_samples_column():
    return get_recipe_config.get("samples_column")

def load_severity():
    severity = {}
    severity["typos_severity"]=get_recipe_config.get("typos_severity")
    severity["word_swap_severity"]=get_recipe_config.get("word_swap_severity")
    severity["word_deletion_severity"]=get_recipe_config.get("word_deletion_severity")
    return severity


def load_attacks_type():
    typos_shift = get_recipe_config.get("typos")
    word_swap_shift = get_recipe_config.get("word_swap")
    word_deletion_shift = get_recipe_config.get("word_deletion")
    return [prior_shift,typos_shift,word_swap_shift,typos_shift]

            
    
    
    