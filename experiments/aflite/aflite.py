aflite_filter(X, y, train_pct=0.5, ROUNDS=10, FILTER_PER_ROUND=1000, 
                  EXPECTATION_ROUNDS=30):
  filtered_idx = np.arange(len(X))

  for i in range(0, ROUNDS+1):
    print('Remaining data points: %d' % len(filtered_idx))
    X_filter, y_filter = X[filtered_idx], y[filtered_idx]

    accs = []
    predictions = np.zeros((len(filtered_idx), EXPECTATION_ROUNDS))

    for j in range(0, EXPECTATION_ROUNDS):
      shuffled_idx = np.random.permutation(len(filtered_idx))
      split_idx = int(len(shuffled_idx)*train_pct)
      train_idx = shuffled_idx[:split_idx]
      test_idx = shuffled_idx[split_idx:]

      clf = LogisticRegression(
      C=50. / len(X), penalty='l1', solver='saga', tol=0.1, verbose=0,
      )

      # train
      clf.fit(X_filter[train_idx], y_filter[train_idx])

      # predict
      y_pred = clf.predict(X_filter[test_idx])
      acc = np.mean(y_pred == y_filter[test_idx])
      accs.append(acc)

      predictions[test_idx, j] = 2*(y_pred == y_filter[test_idx])-1

    aflite_bias = np.mean(accs)
    print(aflite_bias)

    predictability = predictions.mean(axis=1)
    sorted_idx = np.argsort(predictability)
    if i != ROUNDS+1:
      good_idx = sorted_idx[:-FILTER_PER_ROUND]
      filtered_idx = filtered_idx[good_idx]

  return filtered_idx
