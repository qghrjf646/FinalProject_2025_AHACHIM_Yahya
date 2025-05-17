import pandas as pd
import numpy as np
import tensorflow as tf
import argparse

def ndcg_at_k(recommended_items, ground_truth_items, k):
    def dcg(recs, gt):
        return sum([1 / np.log2(i + 2) if rec in gt else 0 for i, rec in enumerate(recs[:k])])
    ndcgs = []
    for user in user_ids:
        rec_items = recommended_items(user)
        true_items = set(small_df[(small_df['user_id'] == user) & (small_df['relevant'])]['video_id'])
        if not true_items:
            continue
        dcg_val = dcg(rec_items, true_items)
        idcg_val = sum([1 / np.log2(i + 2) for i in range(min(len(true_items), k))])
        ndcgs.append(dcg_val / idcg_val if idcg_val > 0 else 0)
    return np.mean(ndcgs) if ndcgs else 0.0

def evaluate_nn_model(N=10):
    # Load content-based features (video characteristics) for all user-video pairs
    eval_pairs = pd.read_csv('eval_pairs.csv')
    X_features = pd.read_csv('eval_content_features_X.csv', index_col=0)
    global small_df 
    small_df = pd.read_csv('small_matrix_with_relevance.csv')
    global user_ids
    user_ids = small_df['user_id'].unique()

    # Merge features into eval_pairs
    eval_pairs = eval_pairs.merge(X_features, left_on='video_id', right_index=True, how='left')
    # Load trained model
    model = tf.keras.models.load_model('nn_content_model.h5')
    # Predict probabilities for each user-video pair
    X_eval = eval_pairs.drop(['user_id', 'video_id'], axis=1).values.astype(np.float32)
    probs = model.predict(X_eval).flatten()
    eval_pairs['prob'] = probs
    eval_pairs['user_id'] = eval_pairs['user_id'].astype(small_df['user_id'].dtype)
    eval_pairs['video_id'] = eval_pairs['video_id'].astype(small_df['video_id'].dtype)
    small_df['relevant'] = small_df['relevant'].astype(int)
    results = []
    def get_user_recs(user):
        user_df = eval_pairs[eval_pairs['user_id'] == user]
        top_videos = user_df.sort_values('prob', ascending=False)['video_id'].head(N).tolist()
        return top_videos
    for user in user_ids:
        recs = get_user_recs(user)
        gt = set(small_df[(small_df['user_id'] == user) & (small_df['relevant'])]['video_id'])
        hits = set(recs) & gt
        precision = len(hits) / N if N > 0 else 0
        recall = len(hits) / len(gt) if len(gt) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        results.append({'user_id': user, 'precision': precision, 'recall': recall, 'f1': f1})
    ndcg = ndcg_at_k(get_user_recs, small_df, N)
    print(f"Mean Precision@{N}: {np.mean([r['precision'] for r in results]):.4f}")
    print(f"Mean Recall@{N}: {np.mean([r['recall'] for r in results]):.4f}")
    print(f"Mean F1@{N}: {np.mean([r['f1'] for r in results]):.4f}")
    print(f"Mean NDCG@{N}: {ndcg:.4f}")
    pd.DataFrame(results).to_csv('nn_content_results.csv', index=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--N', type=int, default=10, help='Top-N for evaluation')
    args = parser.parse_args()
    evaluate_nn_model(N=args.N)
