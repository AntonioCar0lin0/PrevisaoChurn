import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, LabelEncoder

class ChurnDataProcessor:
    def __init__(self, file_path):
        self.file_path = file_path
        self.df = None
        
        # Atributos para armazenar os transformers (para uso futuro ou inversão)
        self.encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        self.scaler = MinMaxScaler()
        self.label_encoder = LabelEncoder()
        
        # Dados Brutos divididos
        self.train_df = None
        self.validation_df = None
        self.test_df = None

    def load_data(self):
        """Carrega o dataset CSV."""
        self.df = pd.read_csv(self.file_path)
        return self.df

    def split_and_balance(self):
        if self.df is None:
            self.load_data()

        # Separando classes
        classe1 = self.df[self.df['Churn'] == 'No']
        classe2 = self.df[self.df['Churn'] == 'Yes']

        # --- Lógica da Classe 1 (Majoritária) ---
        n = len(classe1)
        n50 = n // 2
        rest_size = n - n50
        n25 = rest_size // 2

        # Slicing Classe 1
        c1_50 = classe1.sample(n=n50, random_state=1)
        rest1 = classe1.drop(c1_50.index)
        c1_25A = rest1.sample(n=n25, random_state=2)
        c1_25B = rest1.drop(c1_25A.index)

        # --- Lógica da Classe 2 (Minoritária) ---
        # Slicing Classe 2 (usando frações conforme notebook)
        c2_50 = classe2.sample(frac=0.5, random_state=3)
        rest2 = classe2.drop(c2_50.index)
        c2_25A = rest2.sample(frac=0.5, random_state=4)
        c2_25B = rest2.drop(c2_25A.index)

        # Oversampling (Balanceamento)
        c2_50_bal = c2_50.sample(n=len(c1_50), replace=True, random_state=5)
        c2_25A_bal = c2_25A.sample(n=len(c1_25A), replace=True, random_state=6)
        c2_25B_bal = c2_25B.sample(n=len(c1_25B), replace=True, random_state=7)

        # --- Montagem dos Sets ---
        
        self.train_df = pd.concat([c1_50, c2_50_bal]).sample(frac=1, random_state=10).reset_index(drop=True)
        self.validation_df = pd.concat([c1_25A, c2_25A_bal]).sample(frac=1, random_state=11).reset_index(drop=True)
        self.test_df = pd.concat([c1_25B, c2_25B]).sample(frac=1, random_state=12).reset_index(drop=True)

    def _remove_outliers_iqr(self, df, y, columns):
        df_clean = df.copy()
        y_clean = y.copy()
        indices_to_drop = []

        print("--- Análise de Outliers (IQR) no Treino ---")
        for col in columns:
            Q1 = df_clean[col].quantile(0.25)
            Q3 = df_clean[col].quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = df_clean[(df_clean[col] < lower_bound) | (df_clean[col] > upper_bound)].index
            indices_to_drop.extend(outliers)
            
            if len(outliers) > 0:
                print(f"Col {col}: {len(outliers)} outliers encontrados.")

        indices_to_drop = list(set(indices_to_drop))
        
        if indices_to_drop:
            print(f"Removendo total de {len(indices_to_drop)} registros...")
            df_clean = df_clean.drop(indices_to_drop).reset_index(drop=True)
            y_clean = y_clean.drop(indices_to_drop).reset_index(drop=True)
        else:
            print("Nenhum outlier estatístico crítico encontrado.")
            
        return df_clean, y_clean

    def transform_features(self):
        """
        Aplica as transformações (Limpeza, Encoding, Scaling)
        Retorna os X e y finais prontos para o modelo.
        """
        # Separar X e y
        X_train = self.train_df.drop(columns=['Churn'])
        y_train = self.train_df['Churn']
        
        X_val = self.validation_df.drop(columns=['Churn'])
        y_val = self.validation_df['Churn']
        
        X_test = self.test_df.drop(columns=['Churn'])
        y_test = self.test_df['Churn']

        # Preencher nulos em TotalCharges
        X_train['TotalCharges'] = X_train['TotalCharges'].fillna(0.0)
        X_val['TotalCharges'] = X_val['TotalCharges'].fillna(0.0)
        X_test['TotalCharges'] = X_test['TotalCharges'].fillna(0.0)

        # Definição de colunas
        cat_cols = X_train.select_dtypes(include=['object']).columns
        real_num_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']

        # Remoção de Outliers 
        X_train, y_train = self._remove_outliers_iqr(X_train, y_train, real_num_cols)

        # 1. OneHotEncoder 
        self.encoder.fit(X_train[cat_cols])
        
        train_cat = pd.DataFrame(self.encoder.transform(X_train[cat_cols]), 
                                 columns=self.encoder.get_feature_names_out(cat_cols), 
                                 index=X_train.index)
        val_cat = pd.DataFrame(self.encoder.transform(X_val[cat_cols]), 
                               columns=self.encoder.get_feature_names_out(cat_cols), 
                               index=X_val.index)
        test_cat = pd.DataFrame(self.encoder.transform(X_test[cat_cols]), 
                                columns=self.encoder.get_feature_names_out(cat_cols), 
                                index=X_test.index)

        # 2. MinMaxScaler 
        self.scaler.fit(X_train[real_num_cols])
        
        train_num = pd.DataFrame(self.scaler.transform(X_train[real_num_cols]), 
                                 columns=real_num_cols, index=X_train.index)
        val_num = pd.DataFrame(self.scaler.transform(X_val[real_num_cols]), 
                               columns=real_num_cols, index=X_val.index)
        test_num = pd.DataFrame(self.scaler.transform(X_test[real_num_cols]), 
                                columns=real_num_cols, index=X_test.index)

        # 3. Coluna SeniorCitizen
        train_senior = X_train[['SeniorCitizen']]
        val_senior = X_val[['SeniorCitizen']]
        test_senior = X_test[['SeniorCitizen']]

        # Concatenação Final
        X_train_final = pd.concat([train_num, train_senior, train_cat], axis=1)
        X_val_final = pd.concat([val_num, val_senior, val_cat], axis=1)
        X_test_final = pd.concat([test_num, test_senior, test_cat], axis=1)

        # Codificação do Target (Y)
        y_train_final = self.label_encoder.fit_transform(y_train)
        y_val_final = self.label_encoder.transform(y_val)
        y_test_final = self.label_encoder.transform(y_test)

        return X_train_final, y_train_final, X_val_final, y_val_final, X_test_final, y_test_final

    def run_pipeline(self):
        """
        Executa todo o fluxo em ordem.
        """
        print("Iniciando Pipeline de Dados...")
        self.split_and_balance()
        print("Divisão e Balanceamento concluídos.")
        return self.transform_features()