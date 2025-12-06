from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
import joblib

def train_model(X, y):
    """Modeli train et."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Qiymətləndir
    y_pred = model.predict(X_test)
    auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
    
    print("AUC:", auc)
    print("\nClassification Report:\n", classification_report(y_test, y_pred))
    
    # Modeli saxla
    joblib.dump(model, 'churn_model.pkl')
    
    return model, X_test, y_test