import pandas as pd
import numpy as np
import pickle
import argparse

# Load evaluation data
eval_pairs = pd.read_csv('eval_pairs.csv')
# Load LogisticRegression model
with open('logreg_model.pkl', 'rb') as f:
    logreg_model = pickle.load(f)
# Load content features for evaluation
X_eval_content = pd.read_csv('eval_content_features_X.csv', index_col=0)

# Generate content-based scores
content_scores = logreg_model.predict_proba(X_eval_content)[:, 1]

# Map content scores to each user-video pair in eval_pairs
video_ids_eval = X_eval_content.index if hasattr(X_eval_content, 'index') else X_eval_content['video_id']
video_score_map = dict(zip(video_ids_eval, content_scores))
eval_pairs['score'] = eval_pairs['video_id'].map(video_score_map)
eval_pairs['score'] = eval_pairs['score'].fillna(0)

def get_top_n_recommendations(user_ids, item_ids, scores, N):
    df = pd.DataFrame({'user_id': user_ids, 'video_id': item_ids, 'score': scores})
    df = df.sort_values(['user_id', 'score'], ascending=[True, False])
    top_n = df.groupby('user_id').head(N)
    return top_n

def precision_at_k(recommended, ground_truth, k):
    precisions = []
    for user in recommended['user_id'].unique():
        rec_items = recommended[recommended['user_id'] == user]['video_id'].head(k).tolist()
        true_items = set(ground_truth[ground_truth['user_id'] == user]['video_id'])
        if not true_items:
            continue
        hit_count = len(set(rec_items) & true_items)
        precisions.append(hit_count / k)
    return np.mean(precisions) if precisions else 0.0

def recall_at_k(recommended, ground_truth, k):
    recalls = []
    for user in recommended['user_id'].unique():
        rec_items = recommended[recommended['user_id'] == user]['video_id'].head(k).tolist()
        true_items = set(ground_truth[ground_truth['user_id'] == user]['video_id'])
        if not true_items:
            continue
        hit_count = len(set(rec_items) & true_items)
        recalls.append(hit_count / len(true_items))
    return np.mean(recalls) if recalls else 0.0

def ndcg_at_k(recommended, ground_truth, k):
    def dcg(recs, gt):
        return sum([1 / np.log2(i + 2) if rec in gt else 0 for i, rec in enumerate(recs[:k])])
    ndcgs = []
    for user in recommended['user_id'].unique():
        rec_items = recommended[recommended['user_id'] == user]['video_id'].tolist()
        true_items = set(ground_truth[ground_truth['user_id'] == user]['video_id'])
        if not true_items:
            continue
        dcg_val = dcg(rec_items, true_items)
        idcg_val = sum([1 / np.log2(i + 2) for i in range(min(len(true_items), k))])
        ndcgs.append(dcg_val / idcg_val if idcg_val > 0 else 0)
    return np.mean(ndcgs) if ndcgs else 0.0

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--N', type=int, default=10, help='Top-N for evaluation')
    args = parser.parse_args()
    N = args.N

    # Load updated ground truth with relevance
    try:
        small_df = pd.read_csv('small_matrix_with_relevance.csv')
    except Exception:
        print('Could not load ground truth small_matrix_with_relevance.csv for evaluation.')
        small_df = None

    # Only consider relevant videos (watch_ratio > 0.9)
    if small_df is not None and 'relevant' in small_df.columns:
        ground_truth = small_df[small_df['relevant']]
    else:
        ground_truth = small_df

    # Combine scores (simple average for now)
    top_n_recs = get_top_n_recommendations(eval_pairs['user_id'], eval_pairs['video_id'], eval_pairs['score'], N)
    top_n_recs.to_csv('recommendations.csv', index=False)
    print(f'Top-{N} recommendations saved to recommendations.csv')

    if ground_truth is not None:
        prec = precision_at_k(top_n_recs, ground_truth, N)
        rec = recall_at_k(top_n_recs, ground_truth, N)
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
        ndcg = ndcg_at_k(top_n_recs, ground_truth, N)
        print(f'Precision@{N}: {prec:.4f}')
        print(f'Recall@{N}: {rec:.4f}')
        print(f'F1@{N}: {f1:.4f}')
        print(f'NDCG@{N}: {ndcg:.4f}')
    else:
        print('Evaluation metrics not computed due to missing ground truth.')

    print('Evaluation complete.')
