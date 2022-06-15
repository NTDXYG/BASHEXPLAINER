from nlgeval import compute_metrics

metrics_dict = compute_metrics(hypothesis="result/test_hyp.csv",
                               references=["result/test_ref.csv"], no_skipthoughts=True,
                               no_glove=True)