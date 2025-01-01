import pandas as pd
import urllib.parse
import whois
import ssl
import socket
import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import os
from sklearn.ensemble import RandomForestClassifier

class PhishingDetector:
    def __init__(self, dataset_path='phishing.csv'):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        dataset_path = os.path.join(script_dir, 'phishing.csv')
        self.data = pd.read_csv(dataset_path)
        self.prepare_data()
        self.train_model()

    def prepare_data(self):
        self.data['status'] = self.data['status'].map({'phishing': 1, 'legitimate': 0})
        self.x = self.data.drop(['status'], axis=1)
        self.y = self.data['status']
        self.x = self.x.dropna(axis=1, how='all')
        
        self.numeric_cols = self.x.select_dtypes(include=['number']).columns
        self.categorical_cols = self.x.select_dtypes(include=['object']).columns
        
        self.imputer = SimpleImputer(strategy='mean')
        X_numeric = pd.DataFrame(self.imputer.fit_transform(self.x[self.numeric_cols]), columns=self.numeric_cols)
        X_categorical = pd.get_dummies(self.x[self.categorical_cols], drop_first=True)
        
        self.X_processed = pd.concat([X_numeric, X_categorical], axis=1)
        self.scalar = StandardScaler()
        self.X_scaled = self.scalar.fit_transform(self.X_processed)

    def train_model(self):
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.X_scaled, self.y, test_size=0.2, random_state=42)
        self.model = RandomForestClassifier()
        self.model.fit(self.x_train, self.y_train)

    def extract_features(self, url):
        features = {}
        try:
            parsed_url = urllib.parse.urlparse(url)
            features['url_len'] = len(url)
            features['hostname_len'] = len(parsed_url.netloc)
            
            try:
                domain_info = whois.whois(parsed_url.netloc)
                if domain_info.creation_date:
                    if isinstance(domain_info.creation_date, list):
                        creation_date = domain_info.creation_date[0]
                    else:
                        creation_date = domain_info.creation_date
                    
                    domain_age = (datetime.datetime.now() - creation_date).days
                    features['domain_age'] = max(0, domain_age) 
                    features['domain_age'] = 0
            except Exception as e:
                features['domain_age'] = 0
            
            try:
                ip_address = socket.gethostbyname(parsed_url.netloc)
                features['is_ip'] = 1 if ip_address == parsed_url.netloc else 0
            except:
                features['is_ip'] = 0
            
            suspicious_words = ['login', 'verify', 'secure', 'account', 'update', 'webscr', 'pay']
            features['suspicious_words'] = sum([url.lower().count(word) for word in suspicious_words])
            
            try:
                context = ssl.create_default_context()
                with socket.create_connection((parsed_url.netloc, 443)) as sock:
                    with context.wrap_socket(sock, server_hostname=parsed_url.netloc) as ssock:
                        cert = ssock.getpeercert()
                        features['ssl_valid'] = 1
                        expiry_date = datetime.datetime.strptime(cert['notAfter'], '%b %d %H:%M:%S %Y %Z')
                        features['cert_expiry_days'] = (expiry_date - datetime.datetime.now()).days
            except:
                features['cert_expiry_days'] = 0
                features['ssl_valid'] = 0
            
            return features
        except Exception as e:
            print(f"Error extracting features: {e}")
            return None

    def predict_phishing(self, url):
        url_features = self.extract_features(url)
        if url_features is None:
            print("Could not extract features from the URL")
            return None
        
        features_df = pd.DataFrame(columns=self.X_processed.columns)
        features_df.loc[0] = 0 
        
        for col, val in url_features.items():
            if col in features_df.columns:
                features_df.loc[0, col] = val
        
        features_scaled = self.scalar.transform(features_df)
        
        prediction = self.model.predict(features_scaled)
        probability = self.model.predict_proba(features_scaled)[0][1]
        
        print(f"Phishing Probability: {probability*100}%")
        print(f"Is Phishing: {bool(prediction[0])}")


if __name__ == '__main__':
    pass