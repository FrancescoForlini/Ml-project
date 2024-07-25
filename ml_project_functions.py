from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
from sklearn.feature_selection import RFECV, SelectKBest, f_classif, SelectFromModel
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_validate, StratifiedKFold
import xgboost as xgb


def dataset_preparation(n_df):
    n_df = n_df[(n_df['G'] >= 200)]  # n_df = n_df[(n_df['To'] - n_df['From'] >= 3)]
    n_df.set_index(pd.Index([i for i in range(n_df.shape[0])]))
    n_df['Hof'] = n_df['Hof'].astype(int)  # n_df.loc[:, 'Hof'] = n_df['Hof'].astype(int)

    if 'PER' in list(n_df.columns):
        n_df = n_df.drop(columns=['Unnamed: 14', 'Unnamed: 16', 'Unnamed: 18', 'Unnamed: 19', 'Unnamed: 21',
                                  'Unnamed: 23', 'Unnamed: 24'])
    else:
        n_df = n_df.drop(columns=['Unnamed: 22', 'Unnamed: 23', 'Unnamed: 29'])

    return n_df.fillna(0).sample(frac=1, random_state=23)


# All or hof
def plotting_dataset(df, columns_map):
    n_df = df.copy()
    size = len(columns_map)

    fig, ax = plt.subplots(size, 1, figsize=(7, 3 * size), constrained_layout=True)
    # width_ratios=[3 for _ in range(size)], height_ratios=[5 for _ in range(size)]

    for i, x_y_list in columns_map.items():
        sub_p = ax if size == 1 else ax[i]
        x_col = x_y_list[0]
        y_col = x_y_list[1]

        sub_p.set_title(f"{x_col} X {y_col}")
        sub_p.set_xlabel(x_col)
        sub_p.set_ylabel(y_col)

        dict = {
            'Not hof': [n_df[x_col][n_df['Hof'] == 0], n_df[y_col][n_df['Hof'] == 0], 'yellow'],
            'Hof': [n_df[x_col][n_df['Hof'] == 1], n_df[y_col][n_df['Hof'] == 1], 'blue'],
            'Mean': [sum(n_df[x_col]) / n_df.shape[0], sum(n_df[y_col]) / n_df.shape[0], 'red']
        }
        for label, list_value in dict.items():
            x_vals = list_value[0]
            y_vals = list_value[1]
            color = list_value[2]
            sub_p.scatter(x_vals, y_vals, color=color, label=label)

        sub_p.legend()

    return fig, ax


# All or hof
def plotting_dataset1(df, x_columns, y_columns):
    colors = ['yellow', 'blue', 'red']
    labels = ['Not hof', 'Hof', 'Mean']
    n_df = df.copy()
    size = len(x_columns)

    fig, ax = plt.subplots(size, 1, figsize=(7, 3 * size), constrained_layout=True)
    # width_ratios=[3 for _ in range(size)], height_ratios=[5 for _ in range(size)]

    for i in range(size):
        sub_p = ax[i]
        x_col = x_columns[i]
        y_col = y_columns[i]

        sub_p.set_title(f"{x_col} X {y_col}")
        sub_p.set_xlabel(x_col)
        sub_p.set_ylabel(y_col)

        n_df_x_list = []
        n_df_y_list = []
        for k in range(0, 2):
            n_df_x_list.append(n_df[x_col][n_df['Hof'] == k])
            n_df_y_list.append(n_df[y_col][[n_df['Hof'] == k]])
        n_df_x_list.append(sum(n_df[x_col]) / n_df.shape[0])
        n_df_y_list.append(sum(n_df[y_col]) / n_df.shape[0])
        for j in range(len(colors)):
            color = colors[j]
            label = labels[j]
            x_vals = n_df_x_list[j]
            y_vals = n_df_y_list[j]
            sub_p.scatter(x_vals, y_vals, color=color, label=label)

        sub_p.legend()

    return fig, ax


# All and hof
def plotting_dataset_all_hof(df_all, df_hof, x_columns, y_columns):
    colors = ['yellow', 'blue', 'red']
    labels = ['Not hof', 'hof', 'mean']
    n_df_all, n_df_hof = df_all.copy(), df_hof.copy()
    n_dfs = [n_df_all, n_df_hof]
    size = len(x_columns) * 2  # one for all and one for hof

    fig, ax = plt.subplots(size, 1, figsize=(7, 3 * size), constrained_layout=True)
    # width_ratios=[3 for _ in range(size)], height_ratios=[5 for _ in range(size)]

    title_des = ['all', 'hof']
    for i in range(len(x_columns)):
        i = i * 2
        for n, n_df in enumerate(n_dfs):  # 2

            sub_p = ax[i + n]
            x_col = x_columns[i + n]
            y_col = y_columns[i + n]

            sub_p.set_title(f"{x_col} X {y_col} {title_des[n]}")
            sub_p.set_xlabel(x_col)
            sub_p.set_ylabel(y_col)

            dict = {
                'Not hof': [n_df[x_col][n_df['Hof'] == 0], n_df[y_col][n_df['Hof'] == 0], 'yellow'],
                'Hof': [n_df[x_col][n_df['Hof'] == 1], n_df[y_col][n_df['Hof'] == 1], 'blue'],
                'Mean': [sum(n_df[x_col]) / n_df.shape[0], sum(n_df[y_col]) / n_df.shape[0], 'red']
            }

            for label, list_values in dict.items():
                x_vals = list_values[0]
                y_vals = list_values[1]
                c = list_values[2]
                sub_p.scatter(x_vals, y_vals, color=c, label=label)

            sub_p.legend()

    plt.show()

    return fig, ax


def feature_selection_rfecv(model, n_df, n_columns):
    X = n_df[n_columns]
    y = n_df['Hof']

    rfecv = RFECV(estimator=model, step=1, cv=StratifiedKFold(n_splits=7), scoring="f1_weighted",
                  min_features_to_select=4,
                  n_jobs=-1)
    rfecv.fit(X, y)

    print(f"Optimal number of features: {rfecv.n_features_}")
    selected_features = list(X.columns[rfecv.support_])  # <class 'pandas.core.indexes.base.Index'>
    print("Selected features:")
    print(f"{selected_features} \n")

    return selected_features


def feature_selection_skb(n_df, n_columns):  # chi2, f_classif, mutual_info_classif
    X = n_df[n_columns]
    y = n_df['Hof']

    selector = SelectKBest(f_classif).fit(X, y)

    indices = selector.get_support(indices=True)
    selected_features = list(X.columns[indices])

    print(f"Optimal number of features: {len(selected_features)}")
    print("Selected features:")
    print(f"{selected_features} \n")

    return selected_features


def feature_selection_sfm(model, n_df, n_columns):
    X = n_df[n_columns]
    y = n_df['Hof']

    selector = SelectFromModel(estimator=model).fit(X, y)

    # mask = selector.get_support() selected_features = list(filter(lambda x: mask[columns.index(x)], columns))
    indices = selector.get_support(indices=True)
    selected_features = [n_columns[index] for index in indices]

    print(f"Optimal number of features: {len(selected_features)}")
    print("Selected features:")
    print(f"{selected_features} \n")

    return selected_features


def cross_validating_model(model, X, y):
    cv_results_model = cross_validate(model, X, y, cv=5, scoring=['precision', 'recall'])
    mean_precision_model = np.mean(cv_results_model['test_precision'])
    mean_recall_model = np.mean(cv_results_model['test_recall'])
    print("Cross validation mean precision on training set: ", mean_precision_model)
    print(f"Cross validation mean recall on training set: {mean_recall_model}\n")


# Plotting features1
def plotting_class_report1(class_report_pl_dict, colors, title):  # I'll pass only the plotting columns
    # class_report_pl_dict = {'TN' : true_negatives_players, 'FN' : false_negatives_players,
    # 'TP', : true_positives_players, 'FP' : false_positives_players}
    # colors = ['yellow', 'pink', 'blue', 'lightcyan']

    feature_columns = list(class_report_pl_dict['TN'].columns)
    fig, ax = plt.subplots(len(feature_columns), 1, figsize=(7, 3 * len(feature_columns)), constrained_layout=True)

    fig.suptitle(title, fontsize=16)
    for i, column in enumerate(feature_columns):  # a plot for each feature
        print("column: ", column)
        sub_plot = ax[i]
        sub_plot.set_xlabel('Id')
        sub_plot.set_ylabel(column)
        sub_plot.set_title('Id x ' + column)
        for j, (key, value) in enumerate(class_report_pl_dict.items()):  # a subplot for each plotting_element
            sub_plot.scatter(class_report_pl_dict[key].index, class_report_pl_dict[key], color=colors[j], label=key)
        ''' In pandas, la prima parentesi quadrata [] viene utilizzata per l'indicizzazione delle colonne, mentre la 
        seconda parentesi quadrata (se presente) viene utilizzata per l'indicizzazione delle righe.'''

        sub_plot.legend()

    plt.show()

    return fig, ax


# Plotting features
def plotting_class_report(class_reports_pl_dict, df_columns, colors, title):
    # class_report_pl_dict = {'TN' : true_negatives_players, 'FN' : false_negatives_players,
    # 'TP', : true_positives_players, 'FP' : false_positives_players}
    # colors = ['yellow', 'pink', 'blue', 'lightcyan']

    fig, ax = plt.subplots(len(df_columns), 1, figsize=(7, 3 * len(df_columns)), constrained_layout=True)

    fig.suptitle(title, fontsize=16)
    print(df_columns)
    for i, column in enumerate(df_columns):  # a plot for each feature (with index on x-coordinate)
        sub_plot = ax[i]
        sub_plot.set_xlabel('Id')
        sub_plot.set_ylabel(column)
        sub_plot.set_title('Id x ' + column)
        for j, (key, value) in enumerate(class_reports_pl_dict.items()):
            sub_plot.scatter(class_reports_pl_dict[key].index, class_reports_pl_dict[key][column], color=colors[j],
                             label=key)

        sub_plot.legend()

    return fig, ax


# Classification statistics and plotting selected features
def evaluating_classification_model(y_p, y_t, df, columns, title, p_show):
    print("Feature Columns (list) used for training and test process: ")
    print(f"{columns}\n")

    actual_negatives = y_t[y_t == 0]
    actual_positives = y_t[y_t == 1]
    predicted_negatives = y_p[y_p == 0]
    predicted_positives = y_p[y_p == 1]
    true_negatives = y_t[(y_p == y_t) & (y_p == 0)]
    false_negatives = y_t[(y_p != y_t) & (y_p == 0)]
    true_positives = y_t[(y_p == y_t) & (y_p == 1)]
    false_positives = y_t[(y_p != y_t) & (y_p == 1)]

    print("Actual negatives: ", actual_negatives.size)
    print("Actual positives: ", actual_positives.size)
    print("Predicted negatives: ", predicted_negatives.size)
    print("Predicted positives: ", predicted_positives.size)
    print("True negatives (how many of the predicts are really negative): ",
          true_negatives.size)
    print("False negatives (predicted negatives that are actual positive): ",
          false_negatives.size)  # FN (predicted negatives that are actual positives)
    print("True positives (how many of the predicts are really positive): ",
          true_positives.size)
    print("False positives (predicted positives that are actual negative): ",
          false_positives.size)  # FP (predicted positives that are actual negatives)

    true_negatives_players = df.loc[true_negatives.index]
    false_negatives_players = df.loc[false_negatives.index]
    true_positives_players = df.loc[true_positives.index]
    false_positives_players = df.loc[false_positives.index]

    print("PLayer not Hof (True Negatives):")
    print(true_negatives_players['Player'])
    print("Players really Hof that predicted not (False Negatives): ")
    print(false_negatives_players['Player'])
    print("Players true hof (True Positives): ")
    print(true_positives_players['Player'])
    print("PLayers not Hof that predicted so (False positives): ")
    print(false_positives_players['Player'])

    class_report = classification_report(y_t, y_p)
    print("Classification Report:")
    print(f"{class_report}\n")

    # labels = ['true negatives', 'false negatives', 'true positives', 'false positives'], df = df[columns]

    class_report_players_dict = {
        'TN': true_negatives_players,
        'FN': false_negatives_players,
        'TP': true_positives_players,
        'FP': false_positives_players
    }

    if p_show:
        print("Ok let's plot it")
        start_plotting_column = 2 - int(len(columns) / 2)  # si possono plottare tutte le feature volendo
        list_plotting_columns = columns[start_plotting_column:] if len(columns) > 10 else columns
        fig, ax = plotting_class_report(class_report_players_dict, list_plotting_columns,
                                        ['yellow', 'pink', 'blue', 'lightcyan'], title)
        return fig, ax, classification_report(y_t, y_p, output_dict=True)

    return None, None, classification_report(y_t, y_p, output_dict=True)


def probability_hof_comparing(model, X_t, n_df):
    print("probability_hof_comparison: ", n_df.columns)
    y_proba = model.predict_proba(X_t)
    y_proba_df = pd.DataFrame(y_proba[:, 1], index=n_df.index, columns=['model_hof prob'])
    y_proba_df['Player'] = n_df['Player']
    y_proba_df['Hof prob'] = n_df['HoF Prob']
    y_proba_df['diff_prob'] = (y_proba_df['model_hof prob'] - y_proba_df['Hof prob']).abs()
    y_proba_diff_df = y_proba_df.loc[y_proba_df['diff_prob'] > 0.15, ['Player', 'model_hof prob', 'Hof prob']]
    print("# players with >0.15 diff. on hof prob. : ", y_proba_diff_df.shape[0])
    print(f"{y_proba_diff_df}\n")

    return y_proba_df, y_proba_diff_df


def training_test_pipeline(model, train_n_df, test_n_df, sel_features, plotted):
    X = train_n_df[sel_features]
    y = train_n_df['Hof']

    cross_validating_model(model, X, y)

    model.fit(X, y)

    X_test = test_n_df[sel_features]
    y_test = test_n_df['Hof']

    y_pred = model.predict(X_test)
    fig, ax, class_p = evaluating_classification_model(y_pred, y_test, test_n_df, list(X_test.columns), 'Prediction',
                                                       plotted)
    y_proba_df, y_proba_diff_df = probability_hof_comparing(model, X_test, test_n_df)
    # # return { 'figure' : fig,
    #          'axes' : ax,
    #          'classes_p' : class_p,
    #          'y_proba' : y_proba_df,
    #          'y_proba_diff' : y_proba_diff_df}


def train_test_xboost_pipeline(train_n_df, test_n_df, sel_features, plotted):
    X_train, y_train = train_n_df[sel_features], train_n_df['Hof']

    X_test, y_test = test_n_df[sel_features], test_n_df['Hof']

    dtrain = xgb.DMatrix(data=X_train, label=y_train)
    dtest = xgb.DMatrix(data=X_test, label=y_test)

    params = {
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'max_depth': 5,
        'eta': 0.1,
        'subsample': 0.7,
        'colsample_bytree': 0.8,
        'gamma': 0,
        'lambda': 1,
        'alpha': 0,
        'seed': 23
    }

    num_rounds = 100
    bst = xgb.train(params, dtrain, num_rounds)

    preds = bst.predict(dtest)
    y_pred = pd.Series([1 if p > 0.5 else 0 for p in preds], index=test_n_df.index)

    fig, ax, class_p = evaluating_classification_model(y_pred, y_test, test_n_df, sel_features, 'Prediction', plotted)
    # y_proba_df, y_proba_diff_df = probability_hof_comparing(bst, X_test, test_n_df)
    # return { 'figure' : fig,
    #          'axes' : ax,
    #          'classes_p' : class_p}






