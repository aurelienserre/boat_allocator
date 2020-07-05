"""Assembles data from different sources into required format for optim algo.

Takes the raw data from all different sources (gform, Amilia, boat list,
people skills), and produces all the data required by the optimization
algorithm in the right format.
"""

import pandas as pd


def amilia_participants(file):
    """Format list of participants.

    file: xls file obtained in Amilia, in "Activités", by selecting the
    month (under summer 2020) and so all the sub-programs, and choosing
    "Opérations" > "Exporter les inscriptions par date".
    """
    used_cols = ["Prénom", "Nom de famille", "Activité",
                 "Adresse électronique du participant"]
    index_col = "Adresse électronique du participant"
    participants = pd.read_excel(file, skiprows=[0], header=1,
                                 usecols=used_cols, index_col=index_col)

    return participants


def gform_prefs(file, corrections, config, group="comp"):
    """Transform output of gform to a usable format.

    The goal is to produce a format easily usable by the optimization
    algorithm.

    file: csv file resulting form the gform,
    group: either "comp" or "rec".
    """
    columns = config[group]["gform_cols"]
    preferences = pd.read_csv(file, skiprows=[0], names=columns)

    # make email corrections
    preferences.replace(to_replace=corrections, inplace=True)

    # need the replace bcause timezone cannot be parsed if not in format
    # +/-HHMM, so we have to add zeros. Attention, the minus sign in the
    # original data is a weird one, not the classical "-".
    preferences.date = pd.to_datetime(
        preferences.date.apply(lambda x: x.replace("UTC−4", "-0400")),
        format="%Y/%m/%d %I:%M:%S %p %z")
    # only keep latest submission for everyone
    preferences = preferences.loc[preferences.groupby('email').date.idxmax()]
    preferences.set_index("email", inplace=True)

    # transform semicolon separated weekdays into one col per day, and 0/1
    # coded preferences, both for first and second choices.
    first_choice = {col: preferences[col].str.get_dummies(sep=";")
                    for col in config[group]["gform_cols"]
                    if col.startswith('pref_')}
    first_choice = pd.concat(first_choice, names=["time", "day"], axis=1)

    second_choice = {col: preferences[col].str.get_dummies(sep=";")
                     for col in config[group]["gform_cols"]
                     if col.startswith('backup_')}
    second_choice = pd.concat(second_choice, names=["time", "day"], axis=1)

    # assemble first and second choices into one dataframe.
    newprefs = pd.concat([first_choice, second_choice],
                         keys=['first', 'second'],
                         names=["pref", "time", "day"], axis=1)

    # day before time in hierarchy
    newprefs = newprefs.reorder_levels(["pref", "day", "time"], axis=1)
    # shorten time codes (am1, am2, am), and put weekdays in right order
    rename_map = {'pref_week_am1': 'am1', 'pref_week_am2': 'am2',
                  'pref_we': 'am',
                  'backup_week_am1': 'am1', 'backup_week_am2': 'am2',
                  'backup_we': 'am',
                  'Monday': 1, 'Tuesday': 2, 'Wednesday': 3, 'Thrusday': 4,
                  'Friday': 5, 'Saturday': 6}
    newprefs.rename(columns=rename_map, inplace=True)
    newprefs = newprefs.sort_index(axis=1)
    rename_map = {1: 'Monday', 2: 'Tuesday', 3: 'Wednesday', 4: 'Thrusday',
                  5: 'Friday', 6: 'Saturday'}
    newprefs.rename(columns=rename_map, inplace=True)

    return newprefs


def registration_check(gform, amilia):
    """Cross check people who registered and who answered the gform.

    gform: preferences dataframe containing people who answered the form
    amilia: dataframe of people registered on amilia

    Returns :
        not_reg: (pd.Index) list of people who answered the gform but did not
            register on amilia
        no_gform: (pd.DataFrame) list of people who registered but did not
            answered the gform (with their detailed info)
        intersection: (pd.DataFrame) people who both answered the gform and
            registered on amilia
    """
    not_reg = gform.loc[gform.index.difference(amilia.index)]
    no_gform = amilia.loc[amilia.index.difference(gform.index)]
    intersection = gform.loc[gform.index.intersection(amilia.index)]

    return not_reg, no_gform, intersection
