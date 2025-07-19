import os
import pandas as pd

if __name__ == "__main__":
    res1 = pd.read_csv("test_predictions_emb.csv") 
    res2 = pd.read_csv("test_predictions_cv.csv")   
    res3 = pd.read_csv("svm_test_predictions.csv")
    
    merged = res1.merge(res2, on='file_name', suffixes=('_emb', '_cv'))
    merged = merged.merge(res3, on='file_name')
    merged.columns = ['file_name', 'Cold_emb', 'Cold_cv', 'Cold_svm']
    
    merged['vote_emb'] = (merged['Cold_emb'] == 'C').astype(int)
    merged['vote_cv'] = (merged['Cold_cv'] == 'C').astype(int)
    merged['vote_svm'] = (merged['Cold_svm'] == 'C').astype(int)
    
    merged['total_votes'] = merged['vote_emb'] + merged['vote_cv'] + merged['vote_svm']
    merged['Cold'] = merged['total_votes'].apply(lambda x: 'C' if x >= 2 else 'NC')
    
    result = merged[['file_name', 'Cold']]
    result.to_csv('ensemble_test_predictions.csv', index=False)
    
    print(f"Voting Results:")
    print(f"Total files: {len(result)}")
    print(f"Predicted as C: {(result['Cold'] == 'C').sum()}")
    print(f"Predicted as NC: {(result['Cold'] == 'NC').sum()}")
    print(f"Saved to: ensemble_test_predictions.csv")
    
    print("\nFirst 10 results:")
    print(result.head(10))