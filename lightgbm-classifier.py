# PREDEDIFNE DF, TARGET COLUMN, OUTPUT COLUMN NAME (dest_column), LIST OF ALL PREDICTORS (predictor_list) AND LIST OF ALL CAT PREDICTORS (cat_feats) AND HYPER-PARAMS (params)
.
def lgb_model(df, target_column, dest_column, predictor_list, cat_feats, params):
    
    x_all = df[predictor_list]
    y_all = df[[target_column]]    
    
    # Train-test split with stratified sampling for the imbalance
    x_train, x_test, y_train, y_test = train_test_split(x_all, y_all, stratify=y_all , test_size=0.2, random_state = 0)
    # get a validation set from the train set
    x_train_0, x_val, y_train_0, y_val = train_test_split(x_train, y_train, stratify=y_train, test_size=0.2, random_state = 0)

    output_columns = predictor_list.copy()
    output_columns.append(target_column)
    output_columns.append(dest_column)
    base = pd.DataFrame(columns = output_columns)

    train_set = lgb.Dataset(x_train, y_train, categorical_feature = cat_feats)
    val_set = lgb.Dataset(x_val, y_val, reference = train_set)
    all_set = lgb.Dataset(x_all, y_all, categorical_feature = cat_feats)
    
    # Used train set that includes validation set to include all data
    m_lgb = lgb.train(params, train_set,
                      early_stopping_rounds = 10, valid_sets = val_set, verbose_eval = 10)

    
    #------ Make Preds --------
    y_pred = m_lgb.predict(x_test, num_iteration=m_lgb.best_iteration)
    y_pred  = pd.DataFrame(y_pred, columns = [dest_column])
    
    
    test = pd.concat([x_test, y_test], axis = 1)
    test = test.reset_index(drop=True)
    pred = pd.concat([test,y_pred], axis=1) 
    
    
    base = base.append(pred)
    
    # visualize importance based on gain
    lgb.plot_importance(m_lgb,figsize=(10,10),
                        grid = False, importance_type = "gain",
                    title = 'Feature Importance')
    return base, m_lgb
