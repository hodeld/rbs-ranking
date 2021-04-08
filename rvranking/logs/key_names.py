LOG_KEYS = [
    'nr',
    'date',
    'comment_new',  # empty
    'comment',
    'comment_predict',

    # scikit results
    'acc_sc',
    'mrr_predictions_av',

    # global vars
    'fake',
    'baseline',
    'sampling method',
    'event_tokens',
    'rv_tokens',
    'relevance',

    # sampling
    'mean_rvs',
    'rvs_tooshort',


    # data size results
    'length orig, prep, train, test',

]


tensorflow_k = [
    # tensorflow
    'list_size',
    'embedding_dimension',
    'loss_function',
    'batch_size',
    'learning_rate',
    'dropout_rate',
    'group_size',
    'num_train_steps',
    'hidden_layers_dims',
    'save_checkpoint_steps',

    # evaluation
    'metric/MRR@ALL',
    'metric/ndcg@1',
    'metric/ndcg@5',
    'loss',
    'labels_mean',
    'logits_mean',

# predictions
    'predictions',

]

baseline_k = [
    # baseline
    'mrr_mean',
    'ndcg1_mean',
]
