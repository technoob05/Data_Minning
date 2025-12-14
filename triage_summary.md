# Model Metrics, Feature List, and is_ghosted Definition

## 1. Model Metrics (`outputs/tables/model_metrics.csv`)

```
accuracy,precision,recall,f1,auc,pr_auc,brier_score,target
0.7567003132613992,0.742998352553542,0.6997672614429791,0.7207351178585697,0.8377498315192266,0.8069531807608875,0.16301646959506372,high_cost
0.7713191785589976,0.4447852760736196,0.23349436392914655,0.30623020063357975,0.657129250649986,0.36197885571352817,0.16780973775390726,ghosting
```

## 2. Danh sách 18 features (tên cột trong `pr_features.parquet`)

1.  agent_encoded
2.  task_type_encoded
3.  task_confidence
4.  num_commits
5.  additions
6.  deletions
7.  changed_files
8.  total_changes
9.  title_len
10. body_len
11. created_hour
12. created_dayofweek
13. is_weekend
14. mentions_tests
15. has_checklist
16. links_issue
17. touches_tests
18. touches_docs

## 3. Định nghĩa `is_ghosted` (trong features.py)

```python
# Target 2: Ghosting (Rejected but had effort)
# Definition: Rejected AND (num_comments > 0 OR num_reviews > 0)
feat_df["effort_score"] = feat_df["num_comments"] + feat_df["num_reviews"]
feat_df["is_ghosted"] = ((feat_df["status"] == "rejected") & (feat_df["effort_score"] > 0)).astype(int)
```
