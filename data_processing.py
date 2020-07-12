"""Assembles data from different sources into required format for optim algo.

Takes the raw data from all different sources (gform, Amilia, boat list,
people list), and produces all the data required by the optimization
algorithm in the right format.
"""

import pandas as pd
import numpy as np
import yaml


days_catdtype = pd.CategoricalDtype(
    categories=[
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thrusday",
        "Friday",
        "Saturday",
        "Sunday",
    ],
    ordered=True,
)

weight_dtype = pd.CategoricalDtype(categories=["H", "MH", "M", "L"], ordered=True)
category_dtype = pd.CategoricalDtype(categories=["advanced", "intermediate", "beginner"], ordered=True)


def people(file):
    """Load people list.

    file: file containing the list of people, with their weight category, skill
    class and group
    """
    people = pd.read_excel(file)
    # drop header rows (at begining of comp, master, rec sections)
    people = people.loc[people["group"].notna()]
    # drop column that we don't use
    people.drop(columns="later on", inplace=True)
    # consider heavy heavy the same as heavy (changes nothing for boat alloc)
    people.replace({"weight": {"HH": "H"}}, inplace=True)
    # weight and category as categorical dtype for better sorting
    people.weight = people.weight.astype(weight_dtype)
    people.category = people.category.astype(category_dtype)

    return people


def amilia_participants(file):
    """Format list of participants.

    file: xls file obtained in Amilia, in "Activités", by selecting the
    month (under summer 2020) and so all the sub-programs, and choosing
    "Opérations" > "Exporter les inscriptions par date".
    """
    used_cols = [
        "Prénom",
        "Nom de famille",
        "Activité",
        "Adresse électronique du participant",
    ]
    index_col = "Adresse électronique du participant"
    participants = pd.read_excel(
        file, skiprows=[0], header=1, usecols=used_cols, index_col=index_col
    )

    return participants


def gform_prefs(file, corrections, gform_cols, rename_map):
    """Transform output of gform to a usable format.

    The goal is to produce a format easily usable by the optimization
    algorithm.

    file: csv file resulting form the gform,
    corrections: dict of {wrong_email: correct_email}
    gform_cols: columns in the csv file with gform ansers
    """
    # from nested dict to list of tuples
    gform_cols = [
        (lvl_one, lvl_two)
        for lvl_one, lvl_two_list in gform_cols.items()
        for lvl_two in lvl_two_list
    ]
    preferences = pd.read_csv(file, skiprows=[0], names=gform_cols)

    # make email corrections
    preferences.replace(to_replace=corrections, inplace=True)

    # need the replace bcause timezone cannot be parsed if not in format
    # +/-HHMM, so we have to add zeros. Attention, the minus sign in the
    # original data is a weird one, not the classical "-".
    preferences.date = pd.to_datetime(
        preferences.date.apply(lambda x: x.replace("UTC−4", "-0400")),
        format="%Y/%m/%d %I:%M:%S %p %z",
    )
    # only keep latest submission for everyone
    preferences = preferences.loc[preferences.groupby("email").date.idxmax()]
    preferences.set_index("email", inplace=True)

    # transform semicolon separated days into one col per day, and 0/1
    # coded preferences, both for first and second choices.
    first_choice = {
        col: preferences.pref[col].str.get_dummies(sep=";")
        for col in preferences.pref.columns
    }
    first_choice = pd.concat(first_choice, names=["time", "day"], axis=1)

    second_choice = {
        col: preferences.backup[col].str.get_dummies(sep=";")
        for col in preferences.backup.columns
    }
    second_choice = pd.concat(second_choice, names=["time", "day"], axis=1)

    # assemble first and second choices into one dataframe.
    newprefs = pd.concat(
        [first_choice, second_choice],
        keys=["first", "second"],
        names=["pref", "time", "day"],
        axis=1,
    )

    # day before time in hierarchy
    newprefs = newprefs.reorder_levels(["pref", "day", "time"], axis=1)
    # set "days" level of columns as CategoricalIndex, so that days are
    # automatically sorted in the right order
    days = pd.CategoricalIndex(newprefs.columns.levels[1], dtype=days_catdtype)
    newprefs.columns.set_levels(days, level="day", inplace=True)
    newprefs = newprefs.sort_index(axis=1)
    # use actual names of times slots
    newprefs.rename(columns=rename_map, inplace=True)

    return newprefs


def intersection(people, amilia, gform, exclude):
    """Cross check that people appear in all information sources.

    people: dataframe of people for who we have skill and weight info
    amilia: dataframe of people registered on amilia
    gform: preferences dataframe containing people who answered the form
    exclude: list of people we won't need to consider
    Returns:

        check: (pd.DataFrame) one columm per source of data, saying if people
            appear in this source or not (not including exculded people)
        people: (pd.DataFrame) people info of the intersection
        amilia: (pd.DataFrame) amilia info of the intersection
        gform: (pd.DataFrame) gform answers of the intersection
    """
    isin_people = pd.Series(index=people.index, data="ok", name="people")
    isin_amilia = pd.Series(index=amilia.index, data="ok", name="amilia")
    isin_gform = pd.Series(index=gform.index, data="ok", name="gform")
    check = pd.concat(
        [amilia[["Prénom", "Nom de famille"]], isin_people, isin_amilia, isin_gform],
        join="outer",
        axis=1,
    )
    # only keep people not in exclude list
    check = check.loc[~check.index.isin(exclude)]

    intersection_index = check.dropna().index
    people = people.loc[intersection_index]
    amilia = amilia.loc[intersection_index]
    gform = gform.loc[intersection_index]

    # replace nan with "missing" storing
    check = check.fillna("missing")
    check.sort_index(inplace=True)

    return check, people, amilia, gform


def boats(file):
    """Load boat list.

    file: file containing the list of available skiffs, with their boat class
    (skills required to row them), and the weight categories they can
    accommodate
    """
    boats = pd.read_excel(file, sep=";", header=[1])
    boats.drop(columns=["owner"], inplace=True)
    boats.set_index("name", inplace=True)
    replacments = {np.nan: 0, "x": 1}
    boats.replace({col: replacments for col in ["L", "M", "MH", "H"]}, inplace=True)

    return boats


def boat_skill_map(file):
    """Load a mapping between boat classes and skills required for this class.

    file: yaml file containing a dict with the mapping {skill: boat_class}
    Returns:
        boat_for_skill: a dict mapping {skill: boat_class}
        skill_for_boat: a dict mapping {boat_class: skill}

    """
    boat_for_skill = yaml.safe_load(file.open("r"))
    skill_for_boat = dict()
    for skill, boatclasses in boat_for_skill.items():
        for bclass in boatclasses:
            skill_for_boat.setdefault(bclass, []).append(skill)

    return boat_for_skill, skill_for_boat
